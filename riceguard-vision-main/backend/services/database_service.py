import logging
from datetime import datetime
from typing import Optional

from backend.core.config import settings

logger = logging.getLogger(__name__)


class DatabaseService:
    def __init__(self) -> None:
        self.client = None
        self.db = None
        self.collection = None
        self.enabled = bool(settings.mongodb_uri)

        if self.enabled:
            try:
                from pymongo import MongoClient
            except ImportError as exc:  # pragma: no cover
                logger.warning("pymongo is not installed; database support is disabled: %s", exc)
                self.enabled = False
                return

            self.client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=3000)
            self.db = self.client[settings.mongodb_db]
            self.collection = self.db[settings.mongodb_collection]

    def store_prediction(self, prediction: dict) -> Optional[dict]:
        if not self.enabled or self.collection is None:
            return None

        document = {
            "disease": prediction.get("disease"),
            "confidence": prediction.get("confidence"),
            "image_name": prediction.get("image_name"),
            "timestamp": datetime.utcnow(),
        }
        result = self.collection.insert_one(document)
        document["id"] = str(result.inserted_id)
        return document


database_service = DatabaseService()
