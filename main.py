from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.api import router as api_router
from typing import List, Optional
from datetime import datetime
from database import MongoDB
from models import ImageAnalysis

app = FastAPI(title="AI Media Backend")

# {{ edit_2 }}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

app.include_router(api_router, prefix="/api")

# Initialize MongoDB connection
mongo_db = MongoDB("mongodb://localhost:27017", "image_analysis_db")

# Dependency to get the MongoDB instance


def get_db():
    return mongo_db


@app.get("/")
async def root():
    return {"message": "Welcome to the AI Media Backend"}

# CRUD API endpoints


@app.post("/image-analysis/", response_model=str)
async def create_image_analysis(image_analysis: ImageAnalysis, db: MongoDB = Depends(get_db)):
    return db.create_image_analysis(image_analysis)


@app.get("/image-analysis/{image_id}", response_model=Optional[ImageAnalysis])
async def read_image_analysis(image_id: str, db: MongoDB = Depends(get_db)):
    image = db.get_image_analysis(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image analysis not found")
    return image


@app.get("/image-analysis/", response_model=List[ImageAnalysis])
async def read_all_image_analyses(db: MongoDB = Depends(get_db)):
    return db.get_all_image_analyses()


@app.put("/image-analysis/{image_id}", response_model=bool)
async def update_image_analysis(image_id: str, image_analysis: ImageAnalysis, db: MongoDB = Depends(get_db)):
    updated = db.update_image_analysis(image_id, image_analysis)
    if not updated:
        raise HTTPException(status_code=404, detail="Image analysis not found")
    return updated


@app.delete("/image-analysis/{image_id}", response_model=bool)
async def delete_image_analysis(image_id: str, db: MongoDB = Depends(get_db)):
    deleted = db.delete_image_analysis(image_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Image analysis not found")
    return deleted

# Additional query endpoints


@app.get("/image-analysis/tag/{tag}", response_model=List[ImageAnalysis])
async def get_images_by_tag(tag: str, db: MongoDB = Depends(get_db)):
    return db.get_images_by_tag(tag)


@app.get("/image-analysis/scene-type/{scene_type}", response_model=List[ImageAnalysis])
async def get_images_by_scene_type(scene_type: str, db: MongoDB = Depends(get_db)):
    return db.get_images_by_scene_type(scene_type)


@app.get("/image-analysis/date-range/", response_model=List[ImageAnalysis])
async def get_images_by_date_range(start_date: datetime, end_date: datetime, db: MongoDB = Depends(get_db)):
    return db.get_images_by_date_range(start_date, end_date)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
