"""add_response_evaluations_table

Revision ID: 8c7d26643b60
Revises: b9f2b9f0d3a5
Create Date: 2026-07-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '8c7d26643b60'
down_revision: Union[str, Sequence[str], None] = 'b9f2b9f0d3a5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('response_evaluations',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('message_id', sa.UUID(), nullable=False),
    sa.Column('conversation_id', sa.UUID(), nullable=False),
    sa.Column('relevancy_score', sa.Float(), nullable=True),
    sa.Column('relevancy_reason', sa.Text(), nullable=True),
    sa.Column('faithfulness_score', sa.Float(), nullable=True),
    sa.Column('faithfulness_reason', sa.Text(), nullable=True),
    sa.Column('coherence_score', sa.Float(), nullable=True),
    sa.Column('coherence_reason', sa.Text(), nullable=True),
    sa.Column('evaluated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['message_id'], ['conversation_messages.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_response_evaluations_message_id'), 'response_evaluations', ['message_id'], unique=False)
    op.create_index(op.f('ix_response_evaluations_conversation_id'), 'response_evaluations', ['conversation_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_response_evaluations_conversation_id'), table_name='response_evaluations')
    op.drop_index(op.f('ix_response_evaluations_message_id'), table_name='response_evaluations')
    op.drop_table('response_evaluations')
