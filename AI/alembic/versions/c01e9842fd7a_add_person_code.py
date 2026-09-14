"""add person numeric code

Revision ID: c01e9842fd7a
Revises: 7c4c8fdc9d21
Create Date: 2026-06-27 02:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c01e9842fd7a"
down_revision: Union[str, Sequence[str], None] = "7c4c8fdc9d21"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE SEQUENCE IF NOT EXISTS persons_person_code_seq")
    op.add_column("persons", sa.Column("person_code", sa.BigInteger(), nullable=True))
    op.execute("UPDATE persons SET person_code = nextval('persons_person_code_seq') WHERE person_code IS NULL")
    op.execute(
        "SELECT setval('persons_person_code_seq', COALESCE((SELECT MAX(person_code) FROM persons), 0) + 1, false)"
    )
    op.alter_column(
        "persons",
        "person_code",
        existing_type=sa.BigInteger(),
        nullable=False,
        server_default=sa.text("nextval('persons_person_code_seq')"),
    )
    op.create_index(op.f("ix_persons_person_code"), "persons", ["person_code"], unique=True)
    op.execute("ALTER SEQUENCE persons_person_code_seq OWNED BY persons.person_code")


def downgrade() -> None:
    op.drop_index(op.f("ix_persons_person_code"), table_name="persons")
    op.drop_column("persons", "person_code")
    op.execute("DROP SEQUENCE IF EXISTS persons_person_code_seq")
