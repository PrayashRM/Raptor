"""
app/models/__init__.py

Re-exports all models so that:
    1. Alembic env.py can import Base.metadata and detect all tables.
    2. Any module can do: from app.models import User, Paper, etc.

IMPORTANT: Every new model MUST be imported here.
If a model is not imported here, Alembic will not detect it
and will not generate a migration for it.
"""

from app.models.base import Base

from app.models.user import User
from app.models.paper import Paper
from app.models.saved_paper import SavedPaper
from app.models.chunk import Chunk
from app.models.chat_session import ChatSession
from app.models.chat_message import ChatMessage
from app.models.notification import Notification
from app.models.collection import Collection
from app.models.collection_paper import CollectionPaper
from app.models.tag import Tag
from app.models.pipeline_trace import PipelineTrace
from app.models.refresh_token import RefreshToken
from app.models.verification_token import VerificationToken

__all__ = [
    "Base",
    "User",
    "Paper",
    "SavedPaper",
    "Chunk",
    "ChatSession",
    "ChatMessage",
    "Notification",
    "Collection",
    "CollectionPaper",
    "Tag",
    "PipelineTrace",
    "RefreshToken",
    "VerificationToken",
]