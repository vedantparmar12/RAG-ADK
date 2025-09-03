"""
Session management system for persistent chat history and context.
Optimized for Windows CPU operation.
"""

import logging
import json
import sqlite3
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Container for a single message"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    message_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {},
            "message_id": self.message_id or str(uuid.uuid4())
        }

@dataclass
class Session:
    """Container for a chat session"""
    session_id: str
    created_at: datetime
    updated_at: datetime
    title: Optional[str]
    messages: List[Message]
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    active: bool = True
    
    def add_message(self, message: Message):
        """Add message to session"""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_context_window(self, max_messages: int = 10) -> List[Message]:
        """Get recent messages for context"""
        return self.messages[-max_messages:] if self.messages else []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "title": self.title,
            "messages": [msg.to_dict() for msg in self.messages],
            "context": self.context,
            "metadata": self.metadata,
            "active": self.active
        }

class SessionManager:
    """
    Manages chat sessions with persistent storage.
    CPU-optimized for Windows.
    """
    
    def __init__(self,
                 storage_dir: Optional[str] = None,
                 max_sessions: int = 100,
                 max_messages_per_session: int = 1000,
                 session_timeout: int = 3600,
                 auto_save: bool = True):
        """
        Initialize session manager.
        
        Args:
            storage_dir: Directory for session storage
            max_sessions: Maximum number of active sessions
            max_messages_per_session: Maximum messages per session
            session_timeout: Session timeout in seconds
            auto_save: Whether to auto-save sessions
        """
        self.max_sessions = max_sessions
        self.max_messages_per_session = max_messages_per_session
        self.session_timeout = session_timeout
        self.auto_save = auto_save
        
        # Setup storage directory
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path.home() / ".rag_sessions"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Database path
        self.db_path = self.storage_dir / "sessions.db"
        
        # Active sessions cache
        self.sessions: Dict[str, Session] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize database
        self._init_database()
        
        # Load recent sessions
        self._load_recent_sessions()
        
        # Start cleanup thread
        if auto_save:
            self._start_auto_save_thread()
        
        logger.info(f"Session manager initialized at {self.storage_dir}")
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                title TEXT,
                context TEXT,
                metadata TEXT,
                active INTEGER DEFAULT 1
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_updated 
            ON sessions(updated_at DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_message_session 
            ON messages(session_id, timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def create_session(self,
                      title: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create new session.
        
        Args:
            title: Optional session title
            context: Initial context
            metadata: Session metadata
            
        Returns:
            Session ID
        """
        with self.lock:
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Create session
            session = Session(
                session_id=session_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                title=title or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                messages=[],
                context=context or {},
                metadata=metadata or {},
                active=True
            )
            
            # Check session limit
            if len(self.sessions) >= self.max_sessions:
                self._evict_oldest_session()
            
            # Store session
            self.sessions[session_id] = session
            
            # Save to database
            self._save_session(session)
            
            logger.info(f"Created session: {session_id}")
            return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session or None
        """
        with self.lock:
            # Check cache first
            if session_id in self.sessions:
                return self.sessions[session_id]
            
            # Try loading from database
            session = self._load_session(session_id)
            if session:
                self.sessions[session_id] = session
                return session
            
            return None
    
    def add_message(self,
                   session_id: str,
                   role: str,
                   content: str,
                   metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Add message to session.
        
        Args:
            session_id: Session ID
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Message ID or None
        """
        with self.lock:
            session = self.get_session(session_id)
            if not session:
                logger.warning(f"Session not found: {session_id}")
                return None
            
            # Check message limit
            if len(session.messages) >= self.max_messages_per_session:
                # Remove oldest message
                session.messages.pop(0)
            
            # Create message
            message = Message(
                role=role,
                content=content,
                timestamp=datetime.now(),
                metadata=metadata,
                message_id=str(uuid.uuid4())
            )
            
            # Add to session
            session.add_message(message)
            
            # Save to database
            if self.auto_save:
                self._save_message(session_id, message)
                self._update_session_timestamp(session_id)
            
            return message.message_id
    
    def get_conversation_history(self,
                                session_id: str,
                                max_messages: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history for session.
        
        Args:
            session_id: Session ID
            max_messages: Maximum messages to return
            
        Returns:
            List of messages
        """
        session = self.get_session(session_id)
        if not session:
            return []
        
        messages = session.messages
        if max_messages:
            messages = messages[-max_messages:]
        
        return [msg.to_dict() for msg in messages]
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get session context for RAG.
        
        Args:
            session_id: Session ID
            
        Returns:
            Context dictionary
        """
        session = self.get_session(session_id)
        if not session:
            return {}
        
        # Build context from recent messages
        recent_messages = session.get_context_window()
        
        context = {
            **session.context,
            "session_id": session_id,
            "message_count": len(session.messages),
            "conversation_summary": self._summarize_conversation(recent_messages),
            "recent_topics": self._extract_topics(recent_messages),
            "user_preferences": self._extract_preferences(session)
        }
        
        return context
    
    def list_sessions(self,
                     active_only: bool = True,
                     limit: int = 20) -> List[Dict[str, Any]]:
        """
        List available sessions.
        
        Args:
            active_only: Only return active sessions
            limit: Maximum sessions to return
            
        Returns:
            List of session summaries
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            query = """
                SELECT session_id, created_at, updated_at, title, active
                FROM sessions
            """
            
            if active_only:
                query += " WHERE active = 1"
            
            query += " ORDER BY updated_at DESC LIMIT ?"
            
            cursor.execute(query, (limit,))
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    "session_id": row[0],
                    "created_at": datetime.fromtimestamp(row[1]).isoformat(),
                    "updated_at": datetime.fromtimestamp(row[2]).isoformat(),
                    "title": row[3],
                    "active": bool(row[4])
                })
            
            conn.close()
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    def update_session_title(self, session_id: str, title: str) -> bool:
        """
        Update session title.
        
        Args:
            session_id: Session ID
            title: New title
            
        Returns:
            Success status
        """
        with self.lock:
            session = self.get_session(session_id)
            if not session:
                return False
            
            session.title = title
            session.updated_at = datetime.now()
            
            if self.auto_save:
                self._save_session(session)
            
            return True
    
    def close_session(self, session_id: str) -> bool:
        """
        Close/deactivate session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Success status
        """
        with self.lock:
            session = self.get_session(session_id)
            if not session:
                return False
            
            session.active = False
            session.updated_at = datetime.now()
            
            # Save and remove from cache
            self._save_session(session)
            
            if session_id in self.sessions:
                del self.sessions[session_id]
            
            logger.info(f"Closed session: {session_id}")
            return True
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete session permanently.
        
        Args:
            session_id: Session ID
            
        Returns:
            Success status
        """
        with self.lock:
            # Remove from cache
            if session_id in self.sessions:
                del self.sessions[session_id]
            
            # Delete from database
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                
                conn.commit()
                conn.close()
                
                logger.info(f"Deleted session: {session_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete session: {e}")
                return False
    
    def _save_session(self, session: Session):
        """Save session to database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.created_at.timestamp(),
                session.updated_at.timestamp(),
                session.title,
                json.dumps(session.context),
                json.dumps(session.metadata),
                int(session.active)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
    
    def _save_message(self, session_id: str, message: Message):
        """Save message to database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO messages VALUES (?, ?, ?, ?, ?, ?)
            """, (
                message.message_id,
                session_id,
                message.role,
                message.content,
                message.timestamp.timestamp(),
                json.dumps(message.metadata) if message.metadata else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
    
    def _load_session(self, session_id: str) -> Optional[Session]:
        """Load session from database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Load session
            cursor.execute("""
                SELECT * FROM sessions WHERE session_id = ?
            """, (session_id,))
            
            session_row = cursor.fetchone()
            if not session_row:
                conn.close()
                return None
            
            # Load messages
            cursor.execute("""
                SELECT * FROM messages 
                WHERE session_id = ? 
                ORDER BY timestamp
            """, (session_id,))
            
            messages = []
            for msg_row in cursor.fetchall():
                message = Message(
                    role=msg_row[2],
                    content=msg_row[3],
                    timestamp=datetime.fromtimestamp(msg_row[4]),
                    metadata=json.loads(msg_row[5]) if msg_row[5] else None,
                    message_id=msg_row[0]
                )
                messages.append(message)
            
            conn.close()
            
            # Create session object
            session = Session(
                session_id=session_row[0],
                created_at=datetime.fromtimestamp(session_row[1]),
                updated_at=datetime.fromtimestamp(session_row[2]),
                title=session_row[3],
                messages=messages,
                context=json.loads(session_row[4]) if session_row[4] else {},
                metadata=json.loads(session_row[5]) if session_row[5] else {},
                active=bool(session_row[6])
            )
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None
    
    def _load_recent_sessions(self):
        """Load recent sessions into cache"""
        sessions = self.list_sessions(active_only=True, limit=10)
        
        for session_info in sessions:
            session = self._load_session(session_info["session_id"])
            if session:
                self.sessions[session.session_id] = session
    
    def _update_session_timestamp(self, session_id: str):
        """Update session timestamp in database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE sessions SET updated_at = ? WHERE session_id = ?
            """, (datetime.now().timestamp(), session_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update session timestamp: {e}")
    
    def _evict_oldest_session(self):
        """Evict oldest session from cache"""
        if not self.sessions:
            return
        
        oldest_id = min(
            self.sessions.keys(),
            key=lambda k: self.sessions[k].updated_at
        )
        
        # Save before evicting
        if self.auto_save:
            self._save_session(self.sessions[oldest_id])
        
        del self.sessions[oldest_id]
        logger.info(f"Evicted session from cache: {oldest_id}")
    
    def _summarize_conversation(self, messages: List[Message]) -> str:
        """Generate conversation summary"""
        if not messages:
            return ""
        
        # Simple summary: topics discussed
        topics = []
        for msg in messages:
            if msg.role == "user":
                # Extract key words (simple approach)
                words = msg.content.lower().split()
                important_words = [w for w in words if len(w) > 5][:3]
                topics.extend(important_words)
        
        return f"Recent topics: {', '.join(set(topics)[:5])}" if topics else ""
    
    def _extract_topics(self, messages: List[Message]) -> List[str]:
        """Extract topics from messages"""
        topics = set()
        
        for msg in messages:
            if msg.role == "user":
                # Simple keyword extraction
                words = msg.content.lower().split()
                topics.update(w for w in words if len(w) > 6)
        
        return list(topics)[:10]
    
    def _extract_preferences(self, session: Session) -> Dict[str, Any]:
        """Extract user preferences from session"""
        preferences = {}
        
        # Analyze message patterns
        if session.messages:
            # Check for preferred response length
            avg_user_length = np.mean([
                len(m.content) for m in session.messages 
                if m.role == "user"
            ]) if session.messages else 100
            
            preferences["prefers_detailed"] = avg_user_length > 200
            preferences["session_duration"] = (
                session.updated_at - session.created_at
            ).total_seconds()
        
        return preferences
    
    def _start_auto_save_thread(self):
        """Start auto-save thread"""
        def auto_save_worker():
            import time
            while True:
                time.sleep(60)  # Save every minute
                self._auto_save_all()
        
        thread = threading.Thread(target=auto_save_worker, daemon=True)
        thread.start()
    
    def _auto_save_all(self):
        """Auto-save all active sessions"""
        with self.lock:
            for session in self.sessions.values():
                if session.active:
                    self._save_session(session)