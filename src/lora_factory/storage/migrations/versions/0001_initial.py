"""Create the initial LoRA Factory schema."""

from __future__ import annotations

from alembic import op

from lora_factory.storage.orm import Base

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
