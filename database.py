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
            return ImageAnalysis(**image_dict)
        return None

    def get_all_image_analyses(self) -> List[ImageAnalysis]:
        return [ImageAnalysis(**image_dict) for image_dict in self.collection.find()]

    # Update
    def update_image_analysis(self, image_id: str, image_analysis: ImageAnalysis) -> bool:
        image_dict = image_analysis.dict()
        image_dict['last_analyzed'] = datetime.utcnow()
        result = self.collection.update_one(
            {"_id": ObjectId(image_id)},
            {"$set": image_dict}
        )
        return result.modified_count > 0

    # Delete
    def delete_image_analysis(self, image_id: str) -> bool:
        result = self.collection.delete_one({"_id": ObjectId(image_id)})
        return result.deleted_count > 0

    # Additional query methods
    def get_images_by_tag(self, tag: str) -> List[ImageAnalysis]:
        return [ImageAnalysis(**image_dict) for image_dict in self.collection.find({"tags": tag})]

    def get_images_by_scene_type(self, scene_type: str) -> List[ImageAnalysis]:
        return [ImageAnalysis(**image_dict) for image_dict in self.collection.find({"scene_type": scene_type})]

    def get_images_by_date_range(self, start_date: datetime, end_date: datetime) -> List[ImageAnalysis]:
        return [ImageAnalysis(**image_dict) for image_dict in self.collection.find({
            "last_analyzed": {
                "$gte": start_date,
                "$lte": end_date
            }
        })]
