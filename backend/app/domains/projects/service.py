from sqlalchemy.ext.asyncio import AsyncSession
from app.domains.projects.models import Project
from app.domains.projects.schemas import ProjectCreate

async def create_project(db: AsyncSession, project_in: ProjectCreate) -> Project:
    project = Project(
        name=project_in.name,
        user_id=project_in.user_id
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project
