"""
Schema 2 CRUD Operations for Augustus Chat

This module provides all database operations for video-scoped conversations
(Schema 2), implementing the write and read paths specified in DEVELOPMENT_PHASES.md
Phase 1.

Key Features:
- Video-scoped conversation management (per user+video)
- Transaction-safe message insertion with parent updates
- Efficient tab rendering with denormalized previews
- Support for both per-video and General tab patterns

Dependencies: database.py models (VideoConversation, ConversationMessage, VideosCatalog)
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc
from sqlalchemy.exc import IntegrityError
import uuid
import logging
import json
import time
import os

from database import (
    VideoConversation,
    ConversationMessage,
    VideosCatalog,
    ResponseEvaluation,
    uuid_default,
)

# Debug logging configuration
DEBUG_LOGGING = os.getenv("DEBUG_LOGGING", "false").lower() == "true"
logger = logging.getLogger(__name__)


# ============================================================================
# WRITE PATH OPERATIONS
# ============================================================================


def get_or_create_conversation(
    db: Session,
    user_id: uuid.UUID,
    video_id: str,
    video_url: str,
    video_title: str
) -> VideoConversation:
    """
    Get existing conversation or create new one for (user, video) pair.
    
    This is the entry point for all per-video tab interactions.
    Uses UNIQUE constraint (user_id, video_id) to ensure one conversation per video.
    
    Args:
        db: Database session
        user_id: User UUID
        video_id: YouTube video ID (11 chars)
        video_url: Full YouTube URL
        video_title: Video title for preview
    
    Returns:
        VideoConversation object (existing or newly created)
    
    Raises:
        ValueError: If video_id is invalid format
    """
    if not video_id or len(video_id) > 11:
        raise ValueError(f"Invalid video_id: {video_id}")
    
    # Try to get existing conversation
    conversation = db.query(VideoConversation).filter(
        VideoConversation.user_id == user_id,
        VideoConversation.video_id == video_id
    ).first()
    
    if conversation:
        return conversation
    
    # Create new conversation
    conversation = VideoConversation(
        conversation_id=uuid_default(),
        user_id=user_id,
        video_id=video_id,
        video_url=video_url,
        video_title=video_title[:500],  # Enforce length limit
        created_at=datetime.utcnow(),
        last_message_at=datetime.utcnow(),
        message_count=0,
        is_pinned=False
    )
    
    try:
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation
    except IntegrityError as e:
        # Handle race condition: another request created the same conversation
        db.rollback()
        # Retry fetch
        conversation = db.query(VideoConversation).filter(
            VideoConversation.user_id == user_id,
            VideoConversation.video_id == video_id
        ).first()
        if conversation:
            return conversation
        # If still fails, re-raise
        raise e


def insert_conversation_turn(
    db: Session,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    video_id: str,
    user_message: str,
    assistant_message: str,
    user_tokens: Optional[int] = None,
    assistant_tokens: Optional[int] = None
) -> Tuple[ConversationMessage, ConversationMessage]:
    """
    Insert a complete turn (user message + assistant reply) in a SINGLE atomic transaction.
    
    This ensures both messages are inserted together or neither is inserted,
    preventing orphaned user messages without assistant replies.
    
    Args:
        db: Database session
        conversation_id: Parent conversation UUID
        user_id: User UUID
        video_id: Video ID
        user_message: User's query/message
        assistant_message: AI assistant's response
        user_tokens: Optional token count for user message
        assistant_tokens: Optional token count for assistant message
    
    Returns:
        Tuple of (user_message_obj, assistant_message_obj)
    
    Raises:
        ValueError: If conversation not found or validation fails
    """
    try:
        # Get parent conversation with row lock (prevents race conditions)
        conversation = db.query(VideoConversation).filter(
            VideoConversation.conversation_id == conversation_id
        ).with_for_update().first()
        
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")
        
        # Compute next message_index (0-based)
        last_msg = db.query(ConversationMessage).filter(
            ConversationMessage.conversation_id == conversation_id
        ).order_by(desc(ConversationMessage.message_index)).first()
        
        next_index = (last_msg.message_index + 1) if last_msg else 0
        
        # Create user message
        user_msg = ConversationMessage(
            id=uuid_default(),
            conversation_id=conversation_id,
            user_id=user_id,
            video_id=video_id,
            message_index=next_index,
            role='user',
            content=user_message,
            content_length=len(user_message),
            tokens_estimate=user_tokens,
            created_at=datetime.utcnow()
        )
        
        # Create assistant message (next index)
        assistant_msg = ConversationMessage(
            id=uuid_default(),
            conversation_id=conversation_id,
            user_id=user_id,
            video_id=video_id,
            message_index=next_index + 1,
            role='assistant',
            content=assistant_message,
            content_length=len(assistant_message),
            tokens_estimate=assistant_tokens,
            created_at=datetime.utcnow()
        )
        
        # Update parent conversation (both messages count)
        conversation.message_count += 2
        conversation.last_message_at = datetime.utcnow()
        
        # Update preview fields (truncate to 200-300 chars per chatSchema.md)
        preview_length = 250  # Middle of 200-300 range
        conversation.last_user_message = user_message[:preview_length]
        conversation.last_assistant_message = assistant_message[:preview_length]
        
        # Add both messages to session
        db.add(user_msg)
        db.add(assistant_msg)
        
        # Single commit for entire turn (atomic operation)
        db.commit()
        
        # Refresh to get database-generated fields
        db.refresh(user_msg)
        db.refresh(assistant_msg)
        
        return user_msg, assistant_msg
    
    except Exception as e:
        db.rollback()  # Now this actually works - rolls back entire turn
        raise e


def insert_response_evaluation(
    db: Session,
    message_id: uuid.UUID,
    conversation_id: uuid.UUID,
    results: Dict[str, Dict[str, Any]]
) -> "ResponseEvaluation":
    """
    Persist G-Eval scoring results for an assistant message.

    Args:
        db: Database session
        message_id: The assistant ConversationMessage.id being scored
        conversation_id: Parent conversation UUID
        results: Dict keyed by metric name ("relevancy", "faithfulness", "coherence"),
                 each value a dict with optional "score" (float) and "reason" (str).
                 A metric missing from results (or with score=None) is stored as NULL,
                 since metrics can fail independently.

    Returns:
        The created ResponseEvaluation row.
    """
    evaluation = ResponseEvaluation(
        id=uuid_default(),
        message_id=message_id,
        conversation_id=conversation_id,
        relevancy_score=results.get("relevancy", {}).get("score"),
        relevancy_reason=results.get("relevancy", {}).get("reason"),
        faithfulness_score=results.get("faithfulness", {}).get("score"),
        faithfulness_reason=results.get("faithfulness", {}).get("reason"),
        coherence_score=results.get("coherence", {}).get("score"),
        coherence_reason=results.get("coherence", {}).get("reason"),
        evaluated_at=datetime.utcnow()
    )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)
    return evaluation


# ============================================================================
# READ PATH OPERATIONS
# ============================================================================


def get_conversation_by_id(
    db: Session,
    conversation_id: uuid.UUID,
    user_id: Optional[uuid.UUID] = None
) -> Optional[VideoConversation]:
    """
    Get a single conversation by ID.
    
    Args:
        db: Database session
        conversation_id: Conversation UUID
        user_id: Optional user UUID for authorization check
    
    Returns:
        VideoConversation object or None if not found
    """
    # Debug logging (Phase 1.4): Log query parameters and execution
    query_start = time.time()
    
    if DEBUG_LOGGING:
        log_data = {
            "event": "get_conversation_by_id",
            "conversation_id": str(conversation_id),
            "user_id_filter": str(user_id) if user_id else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.debug(f"[DB] Query conversation: {json.dumps(log_data)}")
    
    query = db.query(VideoConversation).filter(
        VideoConversation.conversation_id == conversation_id
    )
    
    if user_id:
        query = query.filter(VideoConversation.user_id == user_id)
    
    result = query.first()
    query_time = (time.time() - query_start) * 1000  # Convert to ms
    
    # Debug logging (Phase 1.4): Log query result and performance
    if DEBUG_LOGGING or query_time > 100:  # Log if slow query (>100ms)
        log_data = {
            "event": "get_conversation_by_id_result",
            "conversation_id": str(conversation_id),
            "user_id_filter": str(user_id) if user_id else None,
            "conversation_found": result is not None,
            "query_time_ms": round(query_time, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if result:
            log_data["conversation_user_id"] = str(result.user_id)
            log_data["video_id"] = result.video_id
            log_data["video_title"] = result.video_title[:100] if result.video_title else None  # Truncate for logging
            log_data["message_count"] = result.message_count
        
        if query_time > 100:
            logger.warning(f"[DB] Slow query detected: {json.dumps(log_data)}")
        else:
            logger.debug(f"[DB] Query result: {json.dumps(log_data)}")
    
    return result


def get_conversation_messages(
    db: Session,
    conversation_id: uuid.UUID,
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> List[ConversationMessage]:
    """
    Get messages for a conversation in chronological order.
    
    Returns messages ordered by message_index ASC (chronological).
    Uses composite index (conversation_id, message_index) for performance.
    
    Args:
        db: Database session
        conversation_id: Conversation UUID
        limit: Optional limit for pagination
        offset: Optional offset for pagination (by message_index)
    
    Returns:
        List of ConversationMessage objects in chronological order
    """
    query = db.query(ConversationMessage).filter(
        ConversationMessage.conversation_id == conversation_id
    ).order_by(ConversationMessage.message_index)
    
    if offset is not None:
        query = query.offset(offset)
    
    if limit is not None:
        query = query.limit(limit)
    
    return query.all()


def get_recent_messages_for_context(
    db: Session,
    conversation_id: uuid.UUID,
    limit: int = 10
) -> List[ConversationMessage]:
    """
    Get most recent messages for agent context (LLM history).
    
    Returns messages in chronological order (oldest to newest) for LLM.
    This is used to build conversation history for the agent.
    
    Args:
        db: Database session
        conversation_id: Conversation UUID
        limit: Number of recent messages to retrieve (default: 10)
    
    Returns:
        List of ConversationMessage objects in chronological order
    """
    # Fetch latest messages in descending order
    messages = db.query(ConversationMessage).filter(
        ConversationMessage.conversation_id == conversation_id
    ).order_by(desc(ConversationMessage.message_index)).limit(limit).all()
    
    # Reverse to get chronological order (oldest to newest)
    messages.reverse()
    
    return messages


# ============================================================================
# VIDEOS CATALOG OPERATIONS
# ============================================================================


def get_or_create_catalog_entry(
    db: Session,
    video_id: str,
    video_title: str,
    video_url: str,
    synopsis: str = "",
    topics: List[str] = None,
    channel_title: Optional[str] = None,
    published_at: Optional[datetime] = None,
    embedding_version: Optional[str] = None
) -> VideosCatalog:
    """
    Get existing catalog entry or create new one.
    
    This is used during video embedding to ensure catalog metadata exists.
    
    Args:
        db: Database session
        video_id: YouTube video ID (11 chars, PK)
        video_title: Video title
        video_url: Full YouTube URL
        synopsis: Video synopsis (250-300 chars for General tab routing)
        topics: List of topic keywords
        channel_title: Optional channel name
        published_at: Optional publish date
        embedding_version: Optional embedding model version
    
    Returns:
        VideosCatalog object (existing or newly created)
    """
    if topics is None:
        topics = []
    
    # Try to get existing entry
    catalog = db.query(VideosCatalog).filter(
        VideosCatalog.video_id == video_id
    ).first()
    
    if catalog:
        return catalog
    
    # Create new entry
    catalog = VideosCatalog(
        video_id=video_id,
        video_title=video_title[:500],
        video_url=video_url,
        synopsis=synopsis[:300],  # Enforce length limit
        topics=topics,
        channel_title=channel_title[:300] if channel_title else None,
        published_at=published_at,
        embedding_version=embedding_version,
        last_refreshed_at=datetime.utcnow()
    )
    
    try:
        db.add(catalog)
        db.commit()
        db.refresh(catalog)
        return catalog
    except IntegrityError as e:
        # Handle race condition
        db.rollback()
        catalog = db.query(VideosCatalog).filter(
            VideosCatalog.video_id == video_id
        ).first()
        if catalog:
            return catalog
        raise e


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def serialize_message_for_api(message: ConversationMessage) -> Dict[str, Any]:
    """
    Serialize ConversationMessage object for API response.
    
    Args:
        message: ConversationMessage object
    
    Returns:
        Dictionary with message data for frontend
    """
    return {
        "id": str(message.id),
        "conversation_id": str(message.conversation_id),
        "user_id": str(message.user_id),
        "video_id": message.video_id,
        "message_index": message.message_index,
        "role": message.role,
        "content": message.content,
        "content_length": message.content_length,
        "tokens_estimate": message.tokens_estimate,
        "created_at": message.created_at.isoformat()
    }


# ============================================================================
# OWNERSHIP VERIFICATION UTILITIES (Phase 7)
# ============================================================================


def verify_conversation_ownership(
    db: Session,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID
) -> Tuple[bool, Optional[str]]:
    """
    Verify user owns conversation. Returns (is_owner, error_message)
    
    Phase 7.16: Utility function for ownership verification.
    
    Args:
        db: Database session
        conversation_id: Conversation UUID to check
        user_id: User UUID to verify ownership
    
    Returns:
        Tuple of (is_owner: bool, error_message: Optional[str])
        - (True, None) if user owns the conversation
        - (False, "Conversation not found") if conversation doesn't exist
        - (False, "Access denied: ...") if conversation exists but user doesn't own it
    """
    conversation = db.query(VideoConversation).filter(
        VideoConversation.conversation_id == conversation_id
    ).first()
    
    if not conversation:
        return (False, "Conversation not found")
    
    if conversation.user_id != user_id:
        return (
            False, 
            f"Access denied: Conversation owned by user_id {conversation.user_id}, but current user is {user_id}"
        )
    
    return (True, None)

