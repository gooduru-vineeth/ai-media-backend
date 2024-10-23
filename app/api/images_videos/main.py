from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from .unsplash import search_images

router = APIRouter()

# Mount the templates directory
templates = Jinja2Templates(directory="app/api/images_videos/templates")


@router.get("/search_images")
async def api_search_images(query: str = "nature", format: str = "list"):
    results = search_images(query)
    return {"images": results, "format": format}


@router.get("/image_search", response_class=HTMLResponse)
async def image_search(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
