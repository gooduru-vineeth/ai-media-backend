from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ['GEMINI_API_KEY'])

router = APIRouter()

class StoryPrompt(BaseModel):
    prompt: str

@router.post("/generate-story")
async def generate_story(story_prompt: StoryPrompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        response = model.generate_content(
            f"Generate a short story based on the following prompt: {story_prompt.prompt}",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
            )
        )
        
        return {"story": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating story: {str(e)}")