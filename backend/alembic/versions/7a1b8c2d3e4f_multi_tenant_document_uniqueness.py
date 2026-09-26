"""Multi-Tenant Document Uniqueness

Revision ID: 7a1b8c2d3e4f
Revises: e344111cfeb9
Create Date: 2026-09-14 11:30:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7a1b8c2d3e4f"
down_revision: str | Sequence[str] | None = "e344111cfeb9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema to scope document uniqueness by project."""
    # Drop global unique index on canonical_url
    op.drop_index("ix_documents_canonical_url", table_name="documents")

    # Create composite unique constraint on (project_id, canonical_url)
    op.create_unique_constraint(
        "uix_project_canonical_url", "documents", ["project_id", "canonical_url"]
    )

    # Create non-unique index on canonical_url for efficient lookup
    op.create_index(
        "ix_documents_canonical_url", "documents", ["canonical_url"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema to global canonical_url uniqueness."""
    # Drop non-unique index
    op.drop_index("ix_documents_canonical_url", table_name="documents")

    # Drop composite unique constraint
    op.drop_constraint("uix_project_canonical_url", "documents", type_="unique")

    # Re-create global unique index
    op.create_index(
        "ix_documents_canonical_url", "documents", ["canonical_url"], unique=True
    )
