from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # {{ edit_1 }}
from app.api import router as api_router

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

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Media Backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)