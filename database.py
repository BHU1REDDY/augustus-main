"""
Database configuration and models for the FastAPI authentication system

Setup Instructions:
1. Install PostgreSQL locally:
   - macOS: brew install postgresql@15
   - Ubuntu: sudo apt-get install postgresql postgresql-contrib
   - Windows: Download from https://www.postgresql.org/download/

2. Start PostgreSQL service:
   - macOS: brew services start postgresql@15
   - Ubuntu: sudo systemctl start postgresql
   - Windows: Use PostgreSQL service manager

3. Create the database:
   - psql postgres
   - CREATE DATABASE fastapi_auth;
   - CREATE USER postgres WITH PASSWORD 'your_password';
   - GRANT ALL PRIVILEGES ON DATABASE fastapi_auth TO postgres;

4. Set environment variable in .env file:
   DATABASE_URL=postgresql://postgres:your_password@localhost:5432/fastapi_auth

5. Run migrations:
   alembic upgrade head
"""
import os
from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    CheckConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import uuid
from sqlalchemy.engine import make_url
from sqlalchemy import JSON

# Database URL from environment variable
# MUST be set - no default to prevent deployment errors
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "[ERROR] DATABASE_URL environment variable is required!\n"
        "   Set it in .env file for your database connection.\n"
        "   Example: postgresql://user:pass@host:5432/dbname?sslmode=require"
    )

parsed_url = make_url(DATABASE_URL)
DB_BACKEND = parsed_url.get_backend_name()
IS_SQLITE = DB_BACKEND == "sqlite"

if IS_SQLITE:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    UUID_TYPE = String(36)
    EXTRA_DATA_TYPE = JSON
    SCOPES_TYPE = JSON

    def uuid_default() -> str:
        return str(uuid.uuid4())
else:
    from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY

    # Production-ready engine configuration
    # Optimized for NeonDB and cloud databases with auto-suspend
    IS_PRODUCTION = os.getenv("ENVIRONMENT", "development") == "production"

    engine = create_engine(
        DATABASE_URL,
        echo=not IS_PRODUCTION,  # Only log SQL in development
        pool_size=10,  # Connection pool size
        max_overflow=20,  # Max overflow connections
        pool_pre_ping=True,  # Test connections before use
        pool_recycle=300,  # Recycle connections after 5 minutes
        connect_args={
            "connect_timeout": 10,
            "options": "-c timezone=utc",
        },
    )

    UUID_TYPE = UUID(as_uuid=True)
    EXTRA_DATA_TYPE = JSONB
    SCOPES_TYPE = ARRAY(String)

    def uuid_default():
        return uuid.uuid4()

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()

class User(Base):
    """User model for authentication system"""
    __tablename__ = "users"
    
    id = Column(UUID_TYPE, primary_key=True, default=uuid_default)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    scopes = Column(SCOPES_TYPE, default=lambda: ["read", "write"], nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"


class UserSession(Base):
    """
    User session model for agent conversation context.
    Tracks current video context and session state.
    """
    __tablename__ = "user_sessions"
    
    # Identity
    session_id = Column(UUID_TYPE, primary_key=True, default=uuid_default)
    user_id = Column(UUID_TYPE, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    # Video context tracking (critical for follow-up questions)
    current_video_id = Column(String(100), nullable=True)
    current_video_url = Column(Text, nullable=True)
    
    # Session lifecycle
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, default=lambda: datetime.utcnow() + timedelta(hours=24), nullable=False)
    
    def __repr__(self):
        return f"<UserSession(session_id='{self.session_id}', user_id='{self.user_id}', active={self.is_active})>"


class Message(Base):
    """
    Message model for conversation history (sliding window).
    Maintains last 20 messages per session for agent context.
    """
    __tablename__ = "messages"
    
    # Identity
    id = Column(UUID_TYPE, primary_key=True, default=uuid_default)
    session_id = Column(UUID_TYPE, ForeignKey('user_sessions.session_id', ondelete='CASCADE'), nullable=False)
    
    # Ordering (critical for conversation flow)
    message_index = Column(Integer, nullable=False)
    
    # Message content
    message_type = Column(String(10), nullable=False)  # 'human', 'ai', 'system', 'tool'
    content = Column(Text, nullable=False)
    
    # Context tracking
    video_id = Column(String(100), nullable=True)
    
    # Tool tracking (for debugging)
    tool_name = Column(String(100), nullable=True)
    tool_success = Column(Boolean, nullable=True)
    
    # Flexible metadata (using extra_data to avoid SQLAlchemy reserved word)
    extra_data = Column(EXTRA_DATA_TYPE, default=dict, nullable=False)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            "message_type IN ('human', 'ai', 'system', 'tool')",
            name='check_message_type'
        ),
        CheckConstraint(
            "(message_type = 'tool' AND tool_name IS NOT NULL) OR (message_type != 'tool')",
            name='check_tool_fields'
        ),
    )
    
    def __repr__(self):
        return f"<Message(type='{self.message_type}', index={self.message_index}, session='{self.session_id}')>"

# Dependency to get database session
def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create all tables
def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

# Drop all tables (for testing)
def drop_tables():
    """Drop all database tables"""
    Base.metadata.drop_all(bind=engine)
