"""
MongoDB Connection and Operations Test Script
Tests basic MongoDB functionality at mongodb://localhost:27017
"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from datetime import datetime
import sys


def test_connection(uri="mongodb://localhost:27017"):
    """Test basic MongoDB connection"""
    print("=" * 60)
    print("Testing MongoDB Connection")
    print("=" * 60)
    
    try:
        # Create client with a short timeout for faster feedback
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.admin.command('ping')
        print("✓ Successfully connected to MongoDB!")
        
        # Get server info
        server_info = client.server_info()
        print(f"✓ MongoDB Version: {server_info['version']}")
        
        return client
    
    except ConnectionFailure:
        print("✗ Failed to connect to MongoDB")
        print("  Make sure MongoDB is running at:", uri)
        sys.exit(1)
    
    except ServerSelectionTimeoutError:
        print("✗ Connection timeout - MongoDB server not found")
        print("  Make sure MongoDB is running at:", uri)
        sys.exit(1)
    
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)


def test_database_operations(client):
    """Test database operations"""
    print("\n" + "=" * 60)
    print("Testing Database Operations")
    print("=" * 60)
    
    # Use a test database
    db = client['test_db']
    print(f"✓ Using database: test_db")
    
    # List all databases
    db_list = client.list_database_names()
    print(f"✓ Available databases: {', '.join(db_list)}")
    
    return db


def test_collection_operations(db):
    """Test collection operations"""
    print("\n" + "=" * 60)
    print("Testing Collection Operations")
    print("=" * 60)
    
    # Create/get a test collection
    collection = db['test_collection']
    print("✓ Using collection: test_collection")
    
    # Clear any existing test data
    result = collection.delete_many({})
    if result.deleted_count > 0:
        print(f"✓ Cleared {result.deleted_count} existing test documents")
    
    return collection


def test_insert_operations(collection):
    """Test insert operations"""
    print("\n" + "=" * 60)
    print("Testing Insert Operations")
    print("=" * 60)
    
    # Insert a single document
    test_doc = {
        "name": "Test Document",
        "type": "test",
        "created_at": datetime.now(),
        "count": 1
    }
    
    result = collection.insert_one(test_doc)
    print(f"✓ Inserted single document with ID: {result.inserted_id}")
    
    # Insert multiple documents
    test_docs = [
        {"name": "Document 1", "value": 100, "category": "A"},
        {"name": "Document 2", "value": 200, "category": "B"},
        {"name": "Document 3", "value": 300, "category": "A"},
        {"name": "Document 4", "value": 400, "category": "C"},
    ]
    
    result = collection.insert_many(test_docs)
    print(f"✓ Inserted {len(result.inserted_ids)} documents")
    
    return collection


def test_query_operations(collection):
    """Test query operations"""
    print("\n" + "=" * 60)
    print("Testing Query Operations")
    print("=" * 60)
    
    # Count documents
    count = collection.count_documents({})
    print(f"✓ Total documents in collection: {count}")
    
    # Find all documents
    print("\n  All documents:")
    for doc in collection.find().limit(3):
        print(f"    - {doc.get('name', 'N/A')}: {doc}")
    
    # Find with filter
    category_a_count = collection.count_documents({"category": "A"})
    print(f"\n✓ Documents with category='A': {category_a_count}")
    
    # Find one document
    doc = collection.find_one({"type": "test"})
    if doc:
        print(f"✓ Found test document: {doc['name']}")
    
    # Query with projection (select specific fields)
    print("\n  Documents (names only):")
    for doc in collection.find({}, {"name": 1, "_id": 0}):
        print(f"    - {doc.get('name', 'N/A')}")


def test_update_operations(collection):
    """Test update operations"""
    print("\n" + "=" * 60)
    print("Testing Update Operations")
    print("=" * 60)
    
    # Update one document
    result = collection.update_one(
        {"name": "Document 1"},
        {"$set": {"value": 150, "updated": True}}
    )
    print(f"✓ Updated {result.modified_count} document(s)")
    
    # Update multiple documents
    result = collection.update_many(
        {"category": "A"},
        {"$inc": {"value": 50}}
    )
    print(f"✓ Updated {result.modified_count} documents with category='A'")
    
    # Verify update
    doc = collection.find_one({"name": "Document 1"})
    print(f"✓ Verified: Document 1 value is now {doc['value']}")


def test_delete_operations(collection):
    """Test delete operations"""
    print("\n" + "=" * 60)
    print("Testing Delete Operations")
    print("=" * 60)
    
    # Delete one document
    result = collection.delete_one({"name": "Document 4"})
    print(f"✓ Deleted {result.deleted_count} document")
    
    # Delete multiple documents
    result = collection.delete_many({"category": "A"})
    print(f"✓ Deleted {result.deleted_count} documents with category='A'")
    
    # Count remaining
    remaining = collection.count_documents({})
    print(f"✓ Remaining documents: {remaining}")


def test_indexing(collection):
    """Test index operations"""
    print("\n" + "=" * 60)
    print("Testing Index Operations")
    print("=" * 60)
    
    # Create an index
    collection.create_index("name")
    print("✓ Created index on 'name' field")
    
    # List indexes
    indexes = list(collection.list_indexes())
    print(f"✓ Collection has {len(indexes)} index(es):")
    for idx in indexes:
        print(f"    - {idx['name']}: {idx['key']}")


def cleanup(client, db_name="test_db"):
    """Clean up test data"""
    print("\n" + "=" * 60)
    print("Cleanup")
    print("=" * 60)
    
    response = input("\nDelete test database 'test_db'? (y/n): ").strip().lower()
    if response == 'y':
        client.drop_database(db_name)
        print("✓ Test database deleted")
    else:
        print("✓ Test database kept for inspection")


def main():
    """Main test function"""
    print("\n" + "=" * 60)
    print("MongoDB Test Script")
    print("=" * 60)
    print(f"Target: mongodb://localhost:27017")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Test connection
        client = test_connection()
        
        # Test database operations
        db = test_database_operations(client)
        
        # Test collection operations
        collection = test_collection_operations(db)
        
        # Test CRUD operations
        test_insert_operations(collection)
        test_query_operations(collection)
        test_update_operations(collection)
        test_delete_operations(collection)
        
        # Test indexing
        test_indexing(collection)
        
        # Cleanup
        cleanup(client)
        
        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        if 'client' in locals():
            client.close()
            print("\n✓ MongoDB connection closed")


if __name__ == "__main__":
    main()

