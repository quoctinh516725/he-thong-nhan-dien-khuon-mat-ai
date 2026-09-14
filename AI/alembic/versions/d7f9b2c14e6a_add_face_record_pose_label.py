"""add face record pose label

Revision ID: d7f9b2c14e6a
Revises: c01e9842fd7a
Create Date: 2026-06-27 02:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d7f9b2c14e6a"
down_revision: Union[str, Sequence[str], None] = "c01e9842fd7a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("face_records", sa.Column("pose_label", sa.String(), nullable=True))
    op.create_index(op.f("ix_face_records_pose_label"), "face_records", ["pose_label"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_face_records_pose_label"), table_name="face_records")
    op.drop_column("face_records", "pose_label")
