"""drop_unused_synopsis_preview_column

Revision ID: 5c1789c419a9
Revises: 8c7d26643b60
Create Date: 2026-07-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '5c1789c419a9'
down_revision: Union[str, Sequence[str], None] = '8c7d26643b60'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # synopsis_preview was never written or read anywhere in the codebase - dead column.
    op.drop_column('video_conversations', 'synopsis_preview')


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column('video_conversations', sa.Column('synopsis_preview', sa.Text(), nullable=True))
