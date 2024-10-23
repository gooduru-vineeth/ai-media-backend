import os
from dotenv import load_dotenv
import requests
from typing import List, Dict
from IPython.display import display, Image

UNSPLASH_API_URL = "https://api.unsplash.com/search/photos"
# Replace with your actual Unsplash access key

load_dotenv()
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")


def search_images(query: str, per_page: int = 10) -> List[Dict]:
    """
    Search for images on Unsplash based on the given query.

    Args:
        query (str): The search query.
        per_page (int): Number of images to return per page (default: 10, max: 30).

    Returns:
        List[Dict]: A list of dictionaries containing image information.
    """
    headers = {
        "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"
    }

    params = {
        "query": query,
        # Ensure we don't exceed the maximum allowed per_page value of 30 for Unsplash API
        # https://unsplash.com/documentation#search-photos
        "per_page": min(per_page, 30),
    }

    response = requests.get(UNSPLASH_API_URL, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        images = []

        for result in data["results"]:
            image_info = {
                "id": result["id"],
                "description": result.get("description", ""),
                "alt_description": result.get("alt_description", ""),
                "urls": {
                    "raw": result["urls"]["raw"],
                    "full": result["urls"]["full"],
                    "regular": result["urls"]["regular"],
                    "small": result["urls"]["small"],
                    "thumb": result["urls"]["thumb"],
                },
                "user": {
                    "name": result["user"]["name"],
                    "username": result["user"]["username"],
                },
                "links": {
                    "html": result["links"]["html"],
                    "download": result["links"]["download"],
                },
            }
            images.append(image_info)

        return images
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []
