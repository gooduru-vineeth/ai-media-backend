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
  - `models/`: Database models
  - `schemas/`: Pydantic schemas for request/response models

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`