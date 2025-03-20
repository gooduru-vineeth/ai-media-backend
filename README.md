## This Project was made for [Google Gen AI Hackathon](https://devfolio.co/google-genaiexchange) | Below video is a generated from our project: 
Video Link: https://drive.google.com/file/d/1_mw-jeZ373G7fIsf2zKwztOMYBS2BPm9/view

# AI Media Backend

This is a backend project for AI Media using FastAPI.

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS and Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the server: `uvicorn main:app --reload`

## Project Structure

- `main.py`: The main application file
- `app/`: The main application package
  - `api/`: API routes
  - `core/`: Core functionality and config
    - `embedding.py`: Embedding functionality using Jina AI model
  - `models/`: Database models
  - `schemas/`: Pydantic schemas for request/response models

## Features

- RESTful API endpoints
- Text embedding using Jina AI model

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Usage Example

To generate embeddings for text, you can use the `/embed` endpoint:

## Milvus Integration

This project now includes integration with Milvus for efficient similarity search of image analyses. The Jina embedding model is used to generate embeddings for the image descriptions, tags, and keywords.

### Setup

1. Install Milvus following the official documentation: https://milvus.io/docs/install_standalone-docker.md
2. Make sure Milvus is running before starting the application.

### Usage

- When creating a new image analysis, the data is automatically inserted into both MongoDB and Milvus.
- Use the `/search-similar-images/` endpoint to find similar images based on a text query.
