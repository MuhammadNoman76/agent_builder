"""
MongoDB connection and database management.
"""
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional
from config import settings


class MongoDB:
    """MongoDB connection manager using singleton pattern."""
    
    _instance: Optional["MongoDB"] = None
    _client: Optional[AsyncIOMotorClient] = None
    _database: Optional[AsyncIOMotorDatabase] = None
    
    def __new__(cls) -> "MongoDB":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def connect(self) -> None:
        """Establish connection to MongoDB."""
        if self._client is None:
            self._client = AsyncIOMotorClient(settings.MONGODB_URL)
            self._database = self._client[settings.MONGODB_DB_NAME]
            
            # Create indexes
            await self._create_indexes()
            
            print(f"Connected to MongoDB: {settings.MONGODB_DB_NAME}")
    
    async def _create_indexes(self) -> None:
        """Create necessary indexes for collections."""
        # Users collection indexes
        await self._database.users.create_index("email", unique=True)
        await self._database.users.create_index("username", unique=True)
        
        # Flows collection indexes
        await self._database.flows.create_index("user_id")
        await self._database.flows.create_index("flow_id", unique=True)
        
        # Nodes collection indexes
        await self._database.nodes.create_index("flow_id")
        await self._database.nodes.create_index([("flow_id", 1), ("node_id", 1)], unique=True)
        
        # Edges collection indexes
        await self._database.edges.create_index("flow_id")
        await self._database.edges.create_index([("flow_id", 1), ("edge_id", 1)], unique=True)
    
    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._database = None
            print("Disconnected from MongoDB")
    
    @property
    def database(self) -> AsyncIOMotorDatabase:
        """Get database instance."""
        if self._database is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._database
    
    @property
    def client(self) -> AsyncIOMotorClient:
        """Get client instance."""
        if self._client is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._client


# Dependency for FastAPI
async def get_database() -> AsyncIOMotorDatabase:
    """FastAPI dependency to get database instance."""
    mongodb = MongoDB()
    return mongodb.database
