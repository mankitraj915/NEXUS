"""Search Schema

Revision ID: 992b62380ba8
Revises: efdaf50b91f1
Create Date: 2026-08-20 02:13:55.007922

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "992b62380ba8"
down_revision: str | Sequence[str] | None = "efdaf50b91f1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "search_project_stats",
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("total_documents", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("project_id"),
    )
    op.create_table(
        "search_terms",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("term", sa.String(), nullable=False),
        sa.Column("document_frequency", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "term", name="uix_project_term"),
    )
    op.create_index(
        op.f("ix_search_terms_term"), "search_terms", ["term"], unique=False
    )
    op.create_table(
        "search_document_stats",
        sa.Column("document_version_id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_version_id"], ["document_versions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("document_version_id"),
    )
    op.create_index(
        op.f("ix_search_document_stats_project_id"),
        "search_document_stats",
        ["project_id"],
        unique=False,
    )
    op.create_table(
        "search_postings",
        sa.Column("term_id", sa.Uuid(), nullable=False),
        sa.Column("document_version_id", sa.UUID(), nullable=False),
        sa.Column("term_frequency", sa.Integer(), nullable=False),
        sa.Column("positions", postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_version_id"], ["document_versions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["term_id"], ["search_terms.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("term_id", "document_version_id"),
    )


def downgrade() -> None:
    op.drop_table("search_postings")
    op.drop_index(
        op.f("ix_search_document_stats_project_id"), table_name="search_document_stats"
    )
    op.drop_table("search_document_stats")
    op.drop_index(op.f("ix_search_terms_term"), table_name="search_terms")
    op.drop_table("search_terms")
    op.drop_table("search_project_stats")
