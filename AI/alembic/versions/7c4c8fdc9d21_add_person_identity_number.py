"""add person identity number

Revision ID: 7c4c8fdc9d21
Revises: 2b502843068b
Create Date: 2026-06-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "7c4c8fdc9d21"
down_revision: Union[str, Sequence[str], None] = "2b502843068b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("persons", sa.Column("identity_number", sa.String(), nullable=True))
    op.create_index(
        op.f("ix_persons_identity_number"),
        "persons",
        ["identity_number"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_persons_identity_number"), table_name="persons")
    op.drop_column("persons", "identity_number")
