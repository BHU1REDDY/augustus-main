from dotenv import load_dotenv
import os

# Load environment variables FIRST, before any other imports that need them
# Try to load from .env in the same directory as this file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever,EnsembleRetriever,ParentDocumentRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, LLMChainExtractor,LLMListwiseRerank
from langchain.storage import InMemoryStore
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone,ServerlessSpec
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel,HttpUrl
from yt_to_id import get_youtube_video_id
import os,hashlib
from pathlib import Path
#import gradio as gr
from langdetect import detect
from langchain.agents import create_react_agent,AgentExecutor
from langchain import hub
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from fastapi import FastAPI, Path, HTTPException, Query, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
import secrets
import base64
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import get_db, User, UserSession, Message, SessionLocal
from langchain.schema import HumanMessage, AIMessage
try:
    import bcrypt
except ImportError:
    print("Warning: bcrypt not installed. Install with: pip install bcrypt")
    bcrypt = None

from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware

try:
    import jwt
except ImportError:
    print("Warning: PyJWT not installed. Install with: pip install PyJWT")
    jwt = None

app = FastAPI(
    title="FastAPI RAG Application",
    description="YouTube Video Q&A with AI Agent",
    version="1.0.0"
)

# CORS Configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY or SECRET_KEY == "change-me":
    raise ValueError(
        "[ERROR] SECRET_KEY must be set to a strong random value!\n"
        "   Generate one with: openssl rand -hex 32\n"
        "   Set it in .env file or deployment environment"
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# User Storage now handled by PostgreSQL database

# Password hashing utilities
def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    if bcrypt is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="bcrypt not installed. Please install with: pip install bcrypt"
        )
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    if bcrypt is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="bcrypt not installed. Please install with: pip install bcrypt"
        )
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# JWT token utilities
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    if jwt is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PyJWT not installed. Please install with: pip install PyJWT"
        )
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[str]:
    """Verify JWT token and return username"""
    if jwt is None:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except jwt.PyJWTError:
        return None

# OAuth2 Configuration
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "read": "Read access to resources",
        "write": "Write access to resources",
        "admin": "Administrative access"
    }
)

# OAuth2 Authentication Functions
def authenticate_user(username: str, password: str, db: Session):
    """Authenticate user with username and password"""
    # Check if user exists in database
    user = db.query(User).filter(User.username == username).first()
    if user and verify_password(password, user.hashed_password):
        return user
    return None

def get_current_user_oauth2(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Get current user from OAuth2 token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    username = verify_token(token)
    if username is None:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return username

def get_current_user_with_scopes(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Get current user with scopes from OAuth2 token"""
    username = get_current_user_oauth2(token, db)
    user = db.query(User).filter(User.username == username).first()
    return {
        "username": username,
        "scopes": user.scopes if user else []
    }

def require_scope(required_scope: str):
    """Dependency to require specific OAuth2 scope"""
    def scope_checker(current_user: dict = Depends(get_current_user_with_scopes)):
        if required_scope not in current_user["scopes"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not enough permissions. Required scope: {required_scope}"
            )
        return current_user
    return scope_checker

# Legacy Basic Authentication (for backward compatibility)
def get_current_user_basic(username: str = Depends(oauth2_scheme)):
    """Legacy basic authentication - kept for backward compatibility"""
    # This is a placeholder - in practice, you'd implement basic auth here
    # For now, we'll redirect to OAuth2
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Please use OAuth2 authentication. Use /token endpoint to get access token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


class input(BaseModel):
    query:str
    url:Optional[HttpUrl] = None
    session_id: Optional[str] = None  # For session continuity

class AgentResponse(BaseModel):
    answer: str
    session_id: str
    video_context: Optional[str] = None

# User Authentication Models
class UserSignup(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    username: str
    email: str
    created_at: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: list[str] = []

class UserInDB(BaseModel):
    username: str
    email: str
    hashed_password: str
    created_at: str
    is_active: bool = True
    scopes: list[str] = ["read", "write"]  # OAuth2 
    









#---------------------------------------------------------------------------------------------

# Environment variables already loaded at top of file

# Validate required API keys at startup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[WARN] OPENAI_API_KEY not set. OpenAI features will fail.")
    print("   Set OPENAI_API_KEY in .env file to enable AI functionality.")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    print("[WARN] PINECONE_API_KEY not set. Vector search will fail.")
    print("   Set PINECONE_API_KEY in .env file to enable RAG functionality.")

YOUTUBE_PROXY_URL = os.getenv("YOUTUBE_PROXY_URL", "").strip() or None
YOUTUBE_COOKIES_FILE = os.getenv("YOUTUBE_COOKIES_FILE", "").strip() or None
YOUTUBE_COOKIES_HEADER = os.getenv("YOUTUBE_COOKIES_HEADER", "").strip() or None

YOUTUBE_PROXIES = None
if YOUTUBE_PROXY_URL:
    YOUTUBE_PROXIES = {
        "http": YOUTUBE_PROXY_URL,
        "https": YOUTUBE_PROXY_URL,
    }
    print(f"[INFO] Using YouTube proxy endpoint: {YOUTUBE_PROXY_URL}")

YOUTUBE_COOKIES = None
if YOUTUBE_COOKIES_FILE:
    try:
        YOUTUBE_COOKIES = Path(YOUTUBE_COOKIES_FILE).read_text(encoding="utf-8").strip()
        print(f"[INFO] Loaded YouTube cookies from file: {YOUTUBE_COOKIES_FILE}")
    except Exception as e:
        print(f"[WARN] Could not read YouTube cookies file '{YOUTUBE_COOKIES_FILE}': {e}")

if not YOUTUBE_COOKIES and YOUTUBE_COOKIES_HEADER:
    YOUTUBE_COOKIES = YOUTUBE_COOKIES_HEADER
    print("[INFO] Using YouTube cookies provided via environment header")

model = ChatOpenAI(model="gpt-4o-mini")  # Uses OPENAI_API_KEY from environment

# Load agent prompt once at startup (cache it)
try:
    REACT_AGENT_PROMPT = hub.pull("hwchase17/react")
    print("[OK] Agent prompt loaded from LangChain hub")
except Exception as e:
    print(f"[WARN] Could not load agent prompt from hub: {e}")
    print("   Using fallback prompt")
    # Fallback prompt if hub is unavailable
    from langchain.prompts import PromptTemplate
    REACT_AGENT_PROMPT = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

#Pinecone Database setting-----------------------------------------------------------------------------
# Initialize Pinecone client only (indexes created per-user dynamically)
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print("[OK] Pinecone client initialized successfully")
        print("   Indexes will be created per-user on first request")
    except Exception as e:
        print(f"[WARN] Pinecone initialization failed: {e}")
        print("   Pinecone features will not be available.")
        pc = None
else:
    print("[WARN] Pinecone not configured (PINECONE_API_KEY not set)")
    pc = None

parser = StrOutputParser()



#Embedding generation-----------------------
embedding=OpenAIEmbeddings(model='text-embedding-3-large')

# Note: Tools and agent are created per-request inside the endpoint
# This ensures proper context isolation and prevents race conditions




#Fetching video transcript
def fetchTranscript(url:str)->list:
    video_id=get_youtube_video_id(url)
    

    
    api_kwargs = {}
    if YOUTUBE_PROXIES:
        api_kwargs["proxies"] = YOUTUBE_PROXIES
    if YOUTUBE_COOKIES:
        api_kwargs["cookies"] = YOUTUBE_COOKIES

    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id=video_id, **api_kwargs)
        transcript_obj = transcripts.find_transcript(["en", "en-US", "en-GB", "hi"])
        transcript_list = transcript_obj.fetch()
        transcript = " ".join(chunk["text"] for chunk in transcript_list)  # A string is formed from a list of dictionaries
        if detect(transcript) == "hi":
            prompt_translate = PromptTemplate(
                template="<task>\nUse your intelligence to translate this Hindi text to english maintaining semantic meaning, correct english spelling and consistency in translation.</task>\n <context> \n {content}</context> ",
                input_variables=['content']
            )
            chain_translate = prompt_translate | model | parser
            transcript = chain_translate.invoke({'content':transcript})
    except TranscriptsDisabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No captions/subtitles available for this video. Please try a video with captions enabled."
        )
    except NoTranscriptFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No transcript is available for this video in the requested languages."
        )
    except CouldNotRetrieveTranscript as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "YouTube is blocking transcript requests from this server. "
                "Configure a residential proxy via YOUTUBE_PROXY_URL or provide authenticated cookies "
                "via YOUTUBE_COOKIES_FILE / YOUTUBE_COOKIES_HEADER environment variables. "
                f"Original error: {str(e)}"
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch video transcript: {str(e)}"
        )

    l = [transcript, video_id]
    return l


prompt=PromptTemplate(
        template="<role>You are an helpful assistant, You are good in general understanding and working with context.</role>\n <task>Answer the {question} using context</task>\n <instructions>\n1)Query the  vector store to complete the task.\n2)Don't use the internet to answer the question, use only the vector store.\n3)Expand on your answers in simple yet explanatory way.</instructions><context>Here is the necessary info-\n{context}</context>",
        input_variables=['question','context']
)


def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


# Helper function to get or create user-specific Pinecone index
def get_user_index(username: str):
    """
    Get or create a Pinecone index for the specific user.
    Each user gets their own isolated index for embeddings.
    
    Args:
        username: The username to create/get index for
        
    Returns:
        Pinecone Index object for the user
        
    Raises:
        HTTPException: If Pinecone is not available
    """
    if pc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pinecone service is not available. Please configure PINECONE_API_KEY in .env file."
        )
    
    # Sanitize username for Pinecone index naming rules
    # Rules: lowercase, alphanumeric, hyphens only, max 45 chars
    index_name = username.lower()
    index_name = index_name.replace("_", "-").replace(" ", "-")
    # Remove any other special characters
    index_name = ''.join(c for c in index_name if c.isalnum() or c == '-')
    # Ensure it doesn't start/end with hyphen
    index_name = index_name.strip('-')
    # Limit length
    index_name = index_name[:45]
    
    # Check if user's index already exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"[INFO] Creating new Pinecone index for user: {username}")
        print(f"   Index name: {index_name}")
        
        # Create new index for this user
        pc.create_index(
            name=index_name,
            dimension=3072,  # Must match OpenAI embeddings dimension
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        
        print(f"[OK] Index '{index_name}' created successfully")
        print(f"   Note: First request may take ~60 seconds while index initializes")
    else:
        print(f"[INFO] Using existing Pinecone index: {index_name}")
    
    # Return the index handle
    return pc.Index(index_name)


# Health check endpoint (no authentication required)
@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }
    
    # Database check
    try:
        db.execute(text("SELECT 1"))
        health_status["checks"]["database"] = "ok"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["database"] = f"error: {str(e)}"
    
    # Pinecone check
    if pc is not None:
        try:
            pc.list_indexes()
            health_status["checks"]["pinecone"] = "ok"
        except Exception as e:
            health_status["checks"]["pinecone"] = f"error: {str(e)}"
    else:
        health_status["checks"]["pinecone"] = "not_configured"
    
    # OpenAI check (just verify key is set)
    health_status["checks"]["openai"] = "configured" if OPENAI_API_KEY else "not_configured"
    
    return health_status

# User Registration and Authentication Endpoints
@app.post("/signup", response_model=UserResponse)
def signup(user_data: UserSignup, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if username already exists
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hash the password
    hashed_password = hash_password(user_data.password)
    
    # Create new user
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        is_active=True,
        scopes=["read", "write"]  # Default scopes for new users
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return UserResponse(
        username=db_user.username,
        email=db_user.email,
        created_at=db_user.created_at.isoformat()
    )

@app.post("/signin", response_model=TokenResponse)
def signin(user_data: UserLogin, db: Session = Depends(get_db)):
    """Sign in user and return JWT token (legacy endpoint)"""
    # Check if user exists
    user = db.query(User).filter(User.username == user_data.username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Verify password
    if not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Create access token with scopes
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user_data.username,
            "scopes": user.scopes
        }, 
        expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

# OAuth2 Token Endpoint (RFC 6749 compliant)
@app.post("/token", response_model=TokenResponse)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """OAuth2 compliant token endpoint"""
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.username,
            "scopes": user.scopes
        }, 
        expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@app.get("/me", response_model=UserResponse)
def get_current_user_info(current_user: str = Depends(get_current_user_oauth2), db: Session = Depends(get_db)):
    """Get current user information (requires read scope)"""
    user = db.query(User).filter(User.username == current_user).first()
    return UserResponse(
        username=user.username,
        email=user.email,
        created_at=user.created_at.isoformat()
    )

@app.get("/admin/users")
def get_all_users(current_user: dict = Depends(require_scope("admin")), db: Session = Depends(get_db)):
    """Get all users (requires admin scope)"""
    users = db.query(User).all()
    return {
        "users": [
            {
                "username": user.username,
                "email": user.email,
                "created_at": user.created_at.isoformat(),
                "is_active": user.is_active,
                "scopes": user.scopes
            }
            for user in users
        ]
    }

@app.post("/admin/users/{username}/scopes")
def update_user_scopes(
    username: str,
    scopes: list[str],
    current_user: dict = Depends(require_scope("admin")),
    db: Session = Depends(get_db)
):
    """Update user scopes (requires admin scope)"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.scopes = scopes
    db.commit()
    return {"message": f"Scopes updated for user {username}", "scopes": scopes}

#ingesting------------------------------------------------------------------------------------------------
@app.post("/", response_model=AgentResponse)
def answer_from_video(user_input: input, current_user: dict = Depends(require_scope("write")), db: Session = Depends(get_db)) -> dict:
    """
    Intelligent Q&A with agent-based tool selection and persistent context.
    
    Features:
    - User-specific Pinecone indexes for data isolation
    - Session-based conversation history (sliding window)
    - Agent decides between YouTube RAG and web search
    - Supports follow-up questions across requests
    
    Requires 'write' scope.
    """
    # Check if Pinecone client is initialized
    if pc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pinecone service is not available. Please configure PINECONE_API_KEY in .env file."
        )
    
    # ========================================
    # STEP 1: Session Management
    # ========================================
    
    # Get user from database
    user = db.query(User).filter(User.username == current_user["username"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get or create session
    session_id_input = user_input.session_id
    session = None
    
    if session_id_input:
        # Try to find the specific session requested
        try:
            session = db.query(UserSession).filter(
                UserSession.session_id == session_id_input,
                UserSession.user_id == user.id,
                UserSession.is_active == True,
                UserSession.expires_at > datetime.utcnow()
            ).first()
            
            if session:
                print(f"[INFO] Resuming session {session.session_id}")
        except Exception as e:
            print(f"[WARN] Invalid session_id format: {e}")
            session = None
    
    # If no session found, try to auto-resume user's most recent session
    if not session:
        # Try to find user's most recent active session (auto-resume)
        session = db.query(UserSession).filter(
            UserSession.user_id == user.id,
            UserSession.is_active == True,
            UserSession.expires_at > datetime.utcnow()
        ).order_by(UserSession.last_activity.desc()).first()
        
        if session:
            print(f"[INFO] Auto-resumed most recent session {session.session_id}")
        else:
            # No active sessions - create new one
            session = UserSession(user_id=user.id)
            db.add(session)
            db.commit()
            db.refresh(session)
            print(f"[INFO] Created new session {session.session_id} for user {user.username}")
    
    # ========================================
    # STEP 2: Video Embedding (if URL provided)
    # ========================================
    
    query = user_input.query
    url = str(user_input.url) if user_input.url else None
    current_video_id = None
    
    if url:
        try:
            pine_index = get_user_index(user.username)
            
            # Fetch transcript
            l = fetchTranscript(url)
            transcript = l[0]
            video_id = l[1]
            current_video_id = video_id
            
            # Embed in Pinecone
            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
            chunks = parent_splitter.create_documents([transcript])
            
            for i, d in enumerate(chunks):
                d.metadata = {"video_id": video_id, "doc_id": f"{video_id}:p{i}"}
            
            store = PineconeVectorStore(index=pine_index, embedding=embedding)
            ids = [f"{video_id}:{hashlib.sha1(d.page_content.encode('utf-8')).hexdigest()[:16]}" for d in chunks]
            
            # Check if already embedded
            already = False
            if ids:
                probe_ids = ids[:8]
                try:
                    existing = pine_index.fetch(ids=probe_ids)
                    vectors = existing.get("vectors", {}) if isinstance(existing, dict) else getattr(existing, "vectors", {}) or {}
                    already = len(vectors) > 0
                except Exception:
                    already = False
            
            if not already:
                store.add_documents(documents=chunks, ids=ids)
                print(f"[OK] Embedded video {video_id} for user {user.username}")
            else:
                print(f"[INFO] Video {video_id} already embedded")
            
            # Update session with current video
            session.current_video_id = video_id
            session.current_video_url = url
            session.last_activity = datetime.utcnow()
            db.commit()
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    # ========================================
    # STEP 3: Build Agent Context from DB
    # ========================================
    
    # Get recent messages for context
    recent_messages = db.query(Message).filter(
        Message.session_id == session.session_id
    ).order_by(Message.message_index.desc()).limit(10).all()
    recent_messages.reverse()  # Chronological order
    
    # Convert to LangChain format
    history = [
        HumanMessage(content=m.content) if m.message_type == 'human'
        else AIMessage(content=m.content)
        for m in recent_messages
    ]
    
    # ========================================
    # STEP 4: Create Context-Aware Tools
    # ========================================
    
    # Capture context in closure
    session_id_str = str(session.session_id)
    username_str = user.username
    current_vid = current_video_id or session.current_video_id
    
    @tool
    def youtube_rag_tool(question: str) -> str:
        """
        Answers questions about YouTube videos discussed in this conversation.
        Use this when user asks about video content, topics, or specific details from videos.
        Input should be the user's question about the video.
        """
        tool_db = SessionLocal()
        try:
            # Get session context
            sess = tool_db.query(UserSession).filter(
                UserSession.session_id == session_id_str
            ).first()
            
            if not sess or not sess.current_video_id:
                return "No video context available. Please ask the user to provide a YouTube URL."
            
            # Get user's Pinecone index
            pine_index = get_user_index(username_str)
            store = PineconeVectorStore(index=pine_index, embedding=embedding)
            
            # Create retriever with video filter
            retriever = store.as_retriever(
                search_kwargs={
                    'k': 10,
                    'filter': {"video_id": sess.current_video_id}
                }
            )
            
            # Build retrieval chain
            mqr = MultiQueryRetriever.from_llm(retriever=retriever, llm=model)
            compressor = LLMChainExtractor.from_llm(model)
            compression_rtr = ContextualCompressionRetriever(
                base_retriever=mqr,
                base_compressor=compressor
            )
            
            # Build answer chain
            parallel_chain = RunnableParallel({
                'question': RunnablePassthrough(),
                'context': compression_rtr | RunnableLambda(format_docs)
            })
            
            chain = parallel_chain | prompt | model | parser
            answer = chain.invoke(question)
            
            return answer
            
        except Exception as e:
            return f"Error retrieving video information: {str(e)}"
        finally:
            tool_db.close()
    
    # Web search tool
    search_tool = DuckDuckGoSearchRun()
    
    # ========================================
    # STEP 5: Create and Execute Agent
    # ========================================
    
    # Create agent with request-specific tools (use cached prompt)
    agent = create_react_agent(
        llm=model,
        tools=[youtube_rag_tool, search_tool],
        prompt=REACT_AGENT_PROMPT  # Use cached prompt from module level
    )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[youtube_rag_tool, search_tool],
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=5
    )
    
    # Build agent input with context and history
    context_parts = []
    
    # Add conversation history if exists
    if history:
        context_parts.append("Previous conversation:")
        for msg in history[-6:]:  # Last 3 exchanges
            msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
            # Don't truncate - agent needs full context to understand references
            content = msg.content  # Full content for proper context
            context_parts.append(f"{msg_type}: {content}")
        context_parts.append("")  # Empty line
    
    # Add video context
    if current_vid:
        context_parts.append(f"[Just processed YouTube video ID: {current_vid}]")
    elif session.current_video_id:
        context_parts.append(f"[Previously discussed video: {session.current_video_id}]")
    
    context_parts.append(f"\nCurrent question: {query}")
    agent_input = "\n".join(context_parts)
    
    # Execute agent (ReAct doesn't use chat_history parameter)
    try:
        result = agent_executor.invoke({"input": agent_input})
        answer = result["output"]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution error: {str(e)}"
        )
    
    # ========================================
    # STEP 6: Store Conversation in DB
    # ========================================
    
    # Get next message index
    last_msg = db.query(Message).filter(
        Message.session_id == session.session_id
    ).order_by(Message.message_index.desc()).first()
    
    next_index = (last_msg.message_index + 1) if last_msg else 0
    
    # Store user message
    user_msg = Message(
        session_id=session.session_id,
        message_index=next_index,
        message_type='human',
        content=query,
        video_id=current_vid
    )
    db.add(user_msg)
    
    # Store AI response
    ai_msg = Message(
        session_id=session.session_id,
        message_index=next_index + 1,
        message_type='ai',
        content=answer,
        video_id=current_vid
    )
    db.add(ai_msg)
    
    # Commit all changes (trigger will auto-cleanup old messages)
    db.commit()
    
    print(f"[INFO] Stored conversation (messages {next_index}-{next_index+1})")
    
    # ========================================
    # STEP 7: Return Response
    # ========================================
    
    return {
        "answer": answer,
        "session_id": str(session.session_id),
        "video_context": current_vid or session.current_video_id
    }



