"""
Shared MongoDB Utilities
Provides common MongoDB connection and operation functions for the application
"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from datetime import datetime


# MongoDB connection settings
MONGODB_URI = "mongodb://localhost:27017"
DATABASE_NAME = "audio_processing"


def get_mongo_client(uri=MONGODB_URI, timeout_ms=5000):
    """
    Get MongoDB client with connection validation.
    
    Args:
        uri: MongoDB connection URI
        timeout_ms: Server selection timeout in milliseconds
        
    Returns:
        MongoClient: MongoDB client object, or None if connection fails
    """
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        # Test connection
        client.admin.command('ping')
        return client
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"Warning: Could not connect to MongoDB at {uri}: {e}")
        return None
    except Exception as e:
        print(f"Warning: MongoDB connection error: {e}")
        return None


def get_mongo_collection(collection_name, db_name=DATABASE_NAME, create_index=None):
    """
    Get MongoDB collection with optional index creation.
    
    Args:
        collection_name: Name of the collection
        db_name: Name of the database (default: audio_processing)
        create_index: Field name to create an index on (optional)
        
    Returns:
        collection: MongoDB collection object, or None if connection fails
    """
    try:
        client = get_mongo_client()
        if client is None:
            return None
        
        db = client[db_name]
        collection = db[collection_name]
        
        # Create index if specified
        if create_index:
            collection.create_index(create_index, unique=False)
        
        return collection
    except Exception as e:
        print(f"Warning: Could not get MongoDB collection '{collection_name}': {e}")
        return None


def save_to_mongodb(collection_name, document, unique_key=None):
    """
    Save a document to MongoDB with optional upsert on unique key.
    
    Args:
        collection_name: Name of the collection to save to
        document: Dictionary containing the document data
        unique_key: Field name to use for upsert (if None, always insert)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        collection = get_mongo_collection(collection_name, create_index=unique_key)
        if collection is None:
            print(f"Warning: MongoDB not available, skipping save to '{collection_name}'")
            return False
        
        # Add created_at timestamp if not present
        if 'created_at' not in document:
            document['created_at'] = datetime.now()
        
        # Upsert if unique_key specified, otherwise insert
        if unique_key and unique_key in document:
            result = collection.update_one(
                {unique_key: document[unique_key]},
                {'$set': document},
                upsert=True
            )
            print(f"✓ Saved document to MongoDB collection '{collection_name}' (matched: {result.matched_count}, modified: {result.modified_count})")
        else:
            result = collection.insert_one(document)
            print(f"✓ Inserted document to MongoDB collection '{collection_name}' with ID: {result.inserted_id}")
        
        return True
        
    except Exception as e:
        print(f"Warning: Could not save to MongoDB collection '{collection_name}': {e}")
        return False


def load_from_mongodb(collection_name, query=None, projection=None):
    """
    Load documents from MongoDB collection.
    
    Args:
        collection_name: Name of the collection to load from
        query: MongoDB query filter (dict), default {} loads all
        projection: Fields to include/exclude (dict), default None returns all fields
        
    Returns:
        list: List of documents, or empty list if error
    """
    try:
        collection = get_mongo_collection(collection_name)
        if collection is None:
            return []
        
        query = query or {}
        cursor = collection.find(query, projection) if projection else collection.find(query)
        
        return list(cursor)
        
    except Exception as e:
        print(f"Warning: Could not load from MongoDB collection '{collection_name}': {e}")
        return []


def find_one_from_mongodb(collection_name, query):
    """
    Find a single document from MongoDB collection.
    
    Args:
        collection_name: Name of the collection
        query: MongoDB query filter (dict)
        
    Returns:
        dict: Document if found, None otherwise
    """
    try:
        collection = get_mongo_collection(collection_name)
        if collection is None:
            return None
        
        return collection.find_one(query)
        
    except Exception as e:
        print(f"Warning: Could not find document in MongoDB collection '{collection_name}': {e}")
        return None


def delete_from_mongodb(collection_name, query):
    """
    Delete documents from MongoDB collection.
    
    Args:
        collection_name: Name of the collection
        query: MongoDB query filter (dict) for documents to delete
        
    Returns:
        int: Number of documents deleted, or 0 if error
    """
    try:
        collection = get_mongo_collection(collection_name)
        if collection is None:
            return 0
        
        result = collection.delete_many(query)
        print(f"✓ Deleted {result.deleted_count} document(s) from MongoDB collection '{collection_name}'")
        
        return result.deleted_count
        
    except Exception as e:
        print(f"Warning: Could not delete from MongoDB collection '{collection_name}': {e}")
        return 0


def count_documents(collection_name, query=None):
    """
    Count documents in MongoDB collection.
    
    Args:
        collection_name: Name of the collection
        query: MongoDB query filter (dict), default {} counts all
        
    Returns:
        int: Number of documents, or 0 if error
    """
    try:
        collection = get_mongo_collection(collection_name)
        if collection is None:
            return 0
        
        query = query or {}
        return collection.count_documents(query)
        
    except Exception as e:
        print(f"Warning: Could not count documents in MongoDB collection '{collection_name}': {e}")
        return 0

