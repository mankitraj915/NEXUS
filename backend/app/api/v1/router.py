from fastapi import APIRouter
from app.api.v1 import projects, crawl, search

api_router = APIRouter()
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(crawl.router, prefix="/crawl", tags=["Crawl"])
api_router.include_router(search.router, prefix="/search", tags=["Search"])
