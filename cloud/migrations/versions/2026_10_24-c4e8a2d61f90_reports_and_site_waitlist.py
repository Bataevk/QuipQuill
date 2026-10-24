"""reports and site waitlist

Reports from the game (the Report button) and the landing site as a waitlist source.

Revision ID: c4e8a2d61f90
Revises: 7a1f3c9b2e64
Create Date: 2026-10-24 12:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c4e8a2d61f90"
down_revision: str | Sequence[str] | None = "7a1f3c9b2e64"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "reports",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("story_id", sa.String(length=64), nullable=True),
        sa.Column("save_id", sa.String(length=64), nullable=True),
        sa.Column("character", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("transcript", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("client", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_reports_user_id", "reports", ["user_id"])
    op.drop_constraint("waitlist_source_valid", "waitlist", type_="check")
    op.create_check_constraint(
        "waitlist_source_valid",
        "waitlist",
        "source IN ('limit', 'profile', 'onboarding', 'site')",
    )


def downgrade() -> None:
    op.drop_constraint("waitlist_source_valid", "waitlist", type_="check")
    op.execute("DELETE FROM waitlist WHERE source = 'site'")
    op.create_check_constraint(
        "waitlist_source_valid", "waitlist", "source IN ('limit', 'profile', 'onboarding')"
    )
    op.drop_index("ix_reports_user_id", table_name="reports")
    op.drop_table("reports")
