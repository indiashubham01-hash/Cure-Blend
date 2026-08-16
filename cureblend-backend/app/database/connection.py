"""
CureBlend — MongoDB Connection & History CRUD
===============================================
Async MongoDB connection using Motor driver.
Provides CRUD operations for storing and retrieving assessment history.

Falls back gracefully if MongoDB is unreachable (logs warning, continues without persistence).
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from app.config import MONGODB_URL, MONGODB_DB_NAME, MONGODB_COLLECTION_HISTORY


# ══════════════════════════════════════════════════════════════
#  DATABASE CONNECTION MANAGER
# ══════════════════════════════════════════════════════════════

class DatabaseManager:
    """Manages async MongoDB connection lifecycle."""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.collection = None
        self._connected = False
    
    async def connect(self):
        """Establish connection to MongoDB Atlas / local MongoDB."""
        try:
            self.client = AsyncIOMotorClient(
                MONGODB_URL,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=5000
            )
            
            # Verify connection with a ping
            await self.client.admin.command("ping")
            
            self.db = self.client[MONGODB_DB_NAME]
            self.collection = self.db[MONGODB_COLLECTION_HISTORY]
            
            # Create indexes for efficient queries
            await self.collection.create_index("timestamp", background=True)
            await self.collection.create_index("request_id", unique=True, background=True)
            
            self._connected = True
            print(f"  [OK] MongoDB connected: {MONGODB_DB_NAME}.{MONGODB_COLLECTION_HISTORY}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"  [!] MongoDB connection failed: {e}")
            print("  -> Running in offline mode (no history persistence)")
            self._connected = False
        except Exception as e:
            print(f"  [!] MongoDB unexpected error: {e}")
            self._connected = False
    
    async def disconnect(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self._connected = False
            print("  [OK] MongoDB disconnected.")
    
    @property
    def is_connected(self) -> bool:
        return self._connected


# Global singleton
db_manager = DatabaseManager()


# ══════════════════════════════════════════════════════════════
#  CRUD OPERATIONS
# ══════════════════════════════════════════════════════════════

async def save_assessment(record: dict) -> bool:
    """
    Save an assessment record to MongoDB.
    
    Args:
        record: Dictionary containing assessment data (matches PredictionHistoryItem schema)
    
    Returns:
        bool: True if saved successfully, False otherwise
    """
    if not db_manager.is_connected:
        print("  [!] MongoDB not connected. Skipping save.")
        return False
    
    try:
        # Ensure timestamp is present
        if "timestamp" not in record:
            record["timestamp"] = datetime.now(timezone.utc)
        
        # Convert datetime objects to MongoDB-compatible format
        if isinstance(record.get("timestamp"), str):
            record["timestamp"] = datetime.fromisoformat(record["timestamp"])
        
        result = await db_manager.collection.insert_one(record)
        return result.acknowledged
        
    except Exception as e:
        print(f"  [!] Failed to save assessment: {e}")
        return False


async def get_history(
    limit: int = 50,
    skip: int = 0
) -> tuple[list[dict], int]:
    """
    Retrieve assessment history records from MongoDB.
    
    Args:
        limit: Maximum number of records to return
        skip: Number of records to skip (for pagination)
    
    Returns:
        Tuple of (records list, total count)
    """
    if not db_manager.is_connected:
        return [], 0
    
    try:
        # Get total count
        total = await db_manager.collection.count_documents({})
        
        # Fetch records sorted by timestamp descending (newest first)
        cursor = db_manager.collection.find(
            {},
            {"_id": 0}  # Exclude MongoDB internal _id
        ).sort("timestamp", -1).skip(skip).limit(limit)
        
        records = await cursor.to_list(length=limit)
        
        # Convert datetime objects to ISO strings for JSON serialization
        for record in records:
            if isinstance(record.get("timestamp"), datetime):
                record["timestamp"] = record["timestamp"].isoformat()
        
        return records, total
        
    except Exception as e:
        print(f"  [!] Failed to fetch history: {e}")
        return [], 0


async def get_assessment_by_id(request_id: str) -> Optional[dict]:
    """
    Retrieve a single assessment by its request_id.
    
    Args:
        request_id: Unique assessment identifier
    
    Returns:
        Assessment record dict or None
    """
    if not db_manager.is_connected:
        return None
    
    try:
        record = await db_manager.collection.find_one(
            {"request_id": request_id},
            {"_id": 0}
        )
        
        if record and isinstance(record.get("timestamp"), datetime):
            record["timestamp"] = record["timestamp"].isoformat()
        
        return record
        
    except Exception as e:
        print(f"  [!] Failed to fetch assessment {request_id}: {e}")
        return None


async def delete_history() -> int:
    """
    Delete all assessment history (for testing/admin purposes).
    
    Returns:
        Number of deleted records
    """
    if not db_manager.is_connected:
        return 0
    
    try:
        result = await db_manager.collection.delete_many({})
        return result.deleted_count
    except Exception as e:
        print(f"  [!] Failed to delete history: {e}")
        return 0
