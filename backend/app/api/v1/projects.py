from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.domains.projects import schemas, service

router = APIRouter()


@router.post(
    "/", response_model=schemas.ProjectResponse, status_code=status.HTTP_201_CREATED
)
async def create_project(
    project_in: schemas.ProjectCreate, db: AsyncSession = Depends(get_db)
):
    return await service.create_project(db=db, project_in=project_in)
