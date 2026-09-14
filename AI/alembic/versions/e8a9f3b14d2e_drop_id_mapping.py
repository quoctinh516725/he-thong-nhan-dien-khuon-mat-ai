"""drop id mapping

Revision ID: e8a9f3b14d2e
Revises: d7f9b2c14e6a
Create Date: 2026-07-15 14:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e8a9f3b14d2e'
down_revision: Union[str, Sequence[str], None] = 'd7f9b2c14e6a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index(op.f('ix_id_mapping_id'), table_name='id_mapping')
    op.drop_index(op.f('ix_id_mapping_faiss_id'), table_name='id_mapping')
    op.drop_table('id_mapping')


def downgrade() -> None:
    op.create_table('id_mapping',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('person_id', sa.String(), nullable=False),
    sa.Column('faiss_id', sa.BigInteger(), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('person_id')
    )
    op.create_index(op.f('ix_id_mapping_faiss_id'), 'id_mapping', ['faiss_id'], unique=True)
    op.create_index(op.f('ix_id_mapping_id'), 'id_mapping', ['id'], unique=False)
