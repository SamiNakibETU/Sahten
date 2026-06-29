"""Cache des recettes générées par LLM (fallback dernier recours).

Revision ID: 0002_generated_recipes
Revises: 0001_initial
Create Date: 2026-06-29
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0002_generated_recipes"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "generated_recipes",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("dish_norm", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("model", sa.String(64)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_generated_recipes_dish_norm",
        "generated_recipes",
        ["dish_norm"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_generated_recipes_dish_norm", table_name="generated_recipes")
    op.drop_table("generated_recipes")
