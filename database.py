from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from typing import List, Optional
from models import ImageAnalysis  # Update this import

from models import ImageAnalysis


class MongoDB:
    def __init__(self, connection_string: str, database_name: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection = self.db.image_analysis

    # Create
    def create_image_analysis(self, image_analysis: ImageAnalysis) -> str:
        image_dict = image_analysis.dict()
        image_dict['last_analyzed'] = datetime.utcnow()
        result = self.collection.insert_one(image_dict)
        return str(result.inserted_id)

    # Read
    def get_image_analysis(self, image_id: str) -> Optional[ImageAnalysis]:
        image_dict = self.collection.find_one({"_id": ObjectId(image_id)})
        if image_dict:
            return image_dict
        return None

    def get_all_image_analyses(self) -> List[ImageAnalysis]:
        return [image_dict for image_dict in self.collection.find()]

    # Update
    def update_image_analysis(self, image_id: str, image_analysis: ImageAnalysis) -> bool:
        image_dict = image_analysis.dict()
        image_dict['last_analyzed'] = datetime.utcnow()
        result = self.collection.update_one(
            {"_id": ObjectId(image_id)},
            {"$set": image_dict}
        )
        return result.modified_count > 0

    def update_image_analysis_based_on_video_path(self, clip_url: str, data: dict) -> bool:
        result = self.collection.update_one(
            {"metadata.clip_url": clip_url},
            {"$set": data}
        )
        return result.modified_count > 0

    # Delete
    def delete_image_analysis(self, image_id: str) -> bool:
        result = self.collection.delete_one({"_id": ObjectId(image_id)})
        return result.deleted_count > 0

    # Additional query methods
    def get_images_by_tag(self, tag: str) -> List[ImageAnalysis]:
        return [image_dict for image_dict in self.collection.find({"tags": tag})]

    def get_images_by_scene_type(self, scene_type: str) -> List[ImageAnalysis]:
        return [image_dict for image_dict in self.collection.find({"scene_type": scene_type})]

    def get_images_by_date_range(self, start_date: datetime, end_date: datetime) -> List[ImageAnalysis]:
        return [image_dict for image_dict in self.collection.find({
            "last_analyzed": {
                "$gte": start_date,
                "$lte": end_date
            }
        })]


if __name__ == "__main__":
    mongo_db = MongoDB("mongodb://localhost:27017", "image_analysis_db")
    image_analysis = mongo_db.get_image_analysis("66fbac39c7db55262d280788")
    print(image_analysis)
