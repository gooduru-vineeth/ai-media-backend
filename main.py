from app.api.images_videos import images_videos_router
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.api import router as api_router
from typing import List, Optional
from datetime import datetime
from database import MongoDB
from models import ImageAnalysis, create_milvus_collection
from app.core.embedding import jina_embedding_model
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

app = FastAPI(title="AI Media Backend")

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the main API router
app.include_router(api_router, prefix="/api")

# Include the images_videos router
app.include_router(images_videos_router, prefix="/api")

# Initialize MongoDB connection
mongo_db = MongoDB("mongodb://localhost:27017", "image_analysis_db")

# Initialize Milvus collection
milvus_collection = create_milvus_collection()


def setup_milvus_collection():
    # Define the collection name
    collection_name = "image_analysis_2"

    # Check if the collection exists
    if not utility.has_collection(collection_name):
        # Define the schema for the collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64,
                        is_primary=True, auto_id=True),
            # Adjust dim as needed
            FieldSchema(name="image_embedding",
                        dtype=DataType.FLOAT_VECTOR, dim=512)
        ]
        schema = CollectionSchema(fields)

        # Create the collection
        collection = Collection(name=collection_name, schema=schema)
        print(f"Collection '{collection_name}' created.")
    else:
        collection = Collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")

    # Create an index if it doesn't exist
    if not collection.has_index():
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024}
        }
        collection.create_index(
            field_name="image_embedding", index_params=index_params)
        print(f"Index created for collection '{collection_name}'.")
    else:
        print(f"Index already exists for collection '{collection_name}'.")

    # Load the collection
    collection.load()
    print(f"Collection '{collection_name}' loaded successfully.")

    return collection


def insert_document_to_milvus(document):
    # Combine relevant text fields
    text = f"{document['detailed_description']} {' '.join(document['tags'])} {
        ' '.join(document['keywords'])}"

    # Generate embeddings using the Jina embedding model
    doc_embedding = jina_embedding_model.encode([text])[0].tolist()

    # Prepare the entity data for Milvus insertion
    entity = {
        "embedding": doc_embedding,
        "text": text,
        "mongo_id": str(document['_id'])
    }

    # Insert the entity into Milvus
    res = milvus_collection.insert([entity])

    # You might want to log or handle the insertion result
    print(f"Milvus insertion result: {res}")

    return res


def search_similar_documents(query_text, limit=5):
    # Generate embedding for the query text using Jina embedding model
    query_embedding = jina_embedding_model.encode([query_text])[0].tolist()

    # Load the collection into memory for searching
    milvus_collection.load()

    # Define search parameters
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }

    # Perform the search
    results = milvus_collection.search(
        [query_embedding],
        "embedding",
        search_params,
        limit=limit,
        output_fields=["text", "mongo_id"]
    )

    return results[0]

# Dependency to get the MongoDB instance


def get_db():
    return mongo_db


@app.get("/")
async def root():
    return {"message": "Welcome to the AI Media Backend"}

# CRUD API endpoints


@app.post("/image-analysis/", response_model=str)
async def create_image_analysis(image_analysis: ImageAnalysis, db: MongoDB = Depends(get_db)):
    image_id = db.create_image_analysis(image_analysis)
    document = db.get_image_analysis(image_id)
    insert_document_to_milvus(document)
    return image_id


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


@app.post("/search-similar-images/", response_model=List[dict])
async def search_similar_images(query: str, limit: int = 5):
    results = search_similar_documents(query, limit)
    response = []
    for result in results:
        response.append({
            'distance': result.distance,
            'text': result.entity.get('text'),
            'mongo_id': result.entity.get('mongo_id')
        })
    return response


if __name__ == "__main__":
    try:
        milvus_collection = create_milvus_collection()
        if milvus_collection:
            milvus_collection.load()
            print(f"Collection 'image_analysis' loaded successfully.")
        # Your main code here
    except Exception as e:
        print(f"An error occurred: {e}")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
