"""
Backfill G-Eval scores for assistant messages stored before evaluation was
wired into the live chat endpoints.

Usage:
    python scripts/backfill_evaluations.py [--limit N]

No retrieval context is available for historical rows, so Faithfulness is
skipped for backfilled messages (Relevancy and Coherence still run).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import SessionLocal, ConversationMessage
from schema2_crud import insert_response_evaluation
from evaluation import evaluate_response
from database import ResponseEvaluation


def get_unscored_assistant_messages(db, limit: int):
    """Assistant messages with no row yet in response_evaluations."""
    scored_ids = db.query(ResponseEvaluation.message_id).subquery()
    return (
        db.query(ConversationMessage)
        .filter(ConversationMessage.role == 'assistant')
        .filter(~ConversationMessage.id.in_(scored_ids))
        .order_by(ConversationMessage.created_at.asc())
        .limit(limit)
        .all()
    )


def get_preceding_user_message(db, assistant_msg: ConversationMessage):
    return (
        db.query(ConversationMessage)
        .filter(ConversationMessage.conversation_id == assistant_msg.conversation_id)
        .filter(ConversationMessage.message_index == assistant_msg.message_index - 1)
        .filter(ConversationMessage.role == 'user')
        .first()
    )


def main():
    parser = argparse.ArgumentParser(description="Backfill G-Eval scores for historical assistant messages.")
    parser.add_argument("--limit", type=int, default=100, help="Max number of messages to score in this run.")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        messages = get_unscored_assistant_messages(db, args.limit)
        print(f"[BACKFILL] Found {len(messages)} unscored assistant messages")

        scored = 0
        skipped = 0
        for assistant_msg in messages:
            user_msg = get_preceding_user_message(db, assistant_msg)
            if not user_msg:
                print(f"[BACKFILL] Skipping message {assistant_msg.id} - no preceding user message found")
                skipped += 1
                continue

            print(f"[BACKFILL] Scoring message {assistant_msg.id} (conversation {assistant_msg.conversation_id})")
            results = evaluate_response(
                query=user_msg.content,
                answer=assistant_msg.content,
                retrieval_context=[],
            )
            insert_response_evaluation(db, assistant_msg.id, assistant_msg.conversation_id, results)
            scored += 1

        print(f"[BACKFILL] Done. Scored {scored}, skipped {skipped}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
