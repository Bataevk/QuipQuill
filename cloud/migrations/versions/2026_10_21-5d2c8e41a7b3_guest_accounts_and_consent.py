"""guest accounts and consent

Guests of the device (ADR 60): no email and no password; `kind` tells them apart. The accepted
terms are recorded per user.

Revision ID: 5d2c8e41a7b3
Revises: 93c86fa6ef7a
Create Date: 2026-10-21 09:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5d2c8e41a7b3"
down_revision: str | Sequence[str] | None = "93c86fa6ef7a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Several NULL emails do not clash under the existing unique constraint (Postgres).
    op.alter_column("users", "email", existing_type=sa.String(length=320), nullable=True)
    op.alter_column("users", "password_hash", existing_type=sa.Text(), nullable=True)
    op.add_column(
        "users", sa.Column("kind", sa.String(length=8), server_default="email", nullable=False)
    )
    op.add_column("users", sa.Column("terms_version", sa.String(length=32), nullable=True))
    op.add_column(
        "users", sa.Column("terms_accepted_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.create_check_constraint("users_kind_valid", "users", "kind IN ('guest', 'email')")
    op.create_check_constraint(
        "users_email_kind",
        "users",
        "(kind = 'email') = (email IS NOT NULL AND password_hash IS NOT NULL)",
    )


def downgrade() -> None:
    op.execute("DELETE FROM users WHERE kind = 'guest'")
    op.drop_constraint("users_email_kind", "users", type_="check")
    op.drop_constraint("users_kind_valid", "users", type_="check")
    op.drop_column("users", "terms_accepted_at")
    op.drop_column("users", "terms_version")
    op.drop_column("users", "kind")
    op.alter_column("users", "password_hash", existing_type=sa.Text(), nullable=False)
    op.alter_column("users", "email", existing_type=sa.String(length=320), nullable=False)
