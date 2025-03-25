# from langchain_ollama import OllamaEmbeddings
# import pymongo
# from sklearn.decomposition import PCA
# import numpy as np
# from tqdm import tqdm

# # MongoDB connection
# client = pymongo.MongoClient("mongodb+srv://hasanmahmudnayeem3027:JT8J4TYqlCLTVgJu@cluster0.qthig.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
# db = client.sample_mflix
# collection = db.movies

# # Initialize Ollama embeddings
# embeddings = OllamaEmbeddings(model="llama3.2")

# def process_embeddings(limit: int = 1500, target_dim: int = 2048):
#     """
#     Generate, reduce, and store plot embeddings in MongoDB for a limited number of documents.

#     Args:
#         limit (int): Maximum number of documents to process.
#         target_dim (int): Target dimension for embeddings.
#     """
#     # Fetch documents
#     documents = list(collection.find({'plot': {"$exists": True}, 'plot_embedding_langchain': {"$exists": False}}).limit(limit))
#     print(f"Found {len(documents)} documents to process.")

#     if not documents:
#         print("No documents found for processing.")
#         return

#     # Generate embeddings
#     print("Generating embeddings...")
#     embedding_matrix, doc_ids = [], []
#     for doc in tqdm(documents, desc="Processing documents"):
#         try:
#             embedding = embeddings.embed_query(doc['plot'])
#             embedding_matrix.append(embedding)
#             doc_ids.append(doc['_id'])
#         except Exception as e:
#             print(f"Failed to generate embedding for document ID {doc['_id']}: {e}")

#     if not embedding_matrix:
#         print("No embeddings were generated.")
#         return

#     # Pad embedding matrix to ensure enough samples for PCA
#     embedding_matrix = np.array(embedding_matrix)
#     n_samples, n_features = embedding_matrix.shape
#     print(f"Original embedding matrix shape: {embedding_matrix.shape}")

#     if n_samples < target_dim:
#         padding = np.zeros((target_dim - n_samples, n_features))
#         embedding_matrix = np.vstack([embedding_matrix, padding])
#         print(f"Padded embedding matrix shape: {embedding_matrix.shape}")

#     # Reduce dimensions
#     print(f"Reducing dimensions from {n_features} to {target_dim} using PCA...")
#     pca = PCA(n_components=target_dim)
#     reduced_embeddings = pca.fit_transform(embedding_matrix)[:n_samples]  # Only keep the original samples

#     # Update MongoDB
#     print("Updating MongoDB...")
#     for doc_id, embedding in zip(doc_ids, reduced_embeddings):
#         try:
#             collection.update_one({'_id': doc_id}, {'$set': {'plot_embedding_langchain': embedding.tolist()}})
#         except Exception as e:
#             print(f"Failed to update document ID {doc_id}: {e}")

#     print("Finished processing embeddings.")

# # Main logic
# if __name__ == "__main__":
#     process_embeddings(limit=1500, target_dim=2048)

#     # Check how many documents have the reduced embedding field
#     count = collection.count_documents({'plot_embedding_langchain': {'$exists': True}})
#     print(f"Number of documents with 'plot_embedding_langchain' field: {count}")


# ############################################################################################################
# # result = collection.update_many(
# #     {},  # Match all documents
# #     {'$unset': {'plot_embedding_langchain': ""}}  # Remove these fields
# # )

# # # Print the result
# # print(f"Matched documents: {result.matched_count}")
# # print(f"Modified documents: {result.modified_count}")
# ############################################################################################################


# # Count documents with the 'plot_embedding_langchain' field
# count = collection.count_documents({'plot_embedding_langchain': {'$exists': True}})
# print(f"Number of documents with 'plot_embedding_langchain' field: {count}")




from langchain_ollama import OllamaEmbeddings
import pymongo
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchEmbeddingProcessor:
    def __init__(
        self,
        mongodb_uri: str = "mongodb+srv://hasanmahmudnayeem3027:JT8J4TYqlCLTVgJu@cluster0.qthig.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
        database_name: str = "sample_mflix",
        collection_name: str = "movies",
        batch_size: int = 50,
        target_dim: int = 2048
    ):
        self.client = pymongo.MongoClient(mongodb_uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        self.embeddings = OllamaEmbeddings(model="llama3.2")
        self.batch_size = batch_size
        self.target_dim = target_dim

    def get_unprocessed_documents(self, batch_size: int) -> List[Dict[str, Any]]:
        """Fetch a batch of unprocessed documents."""
        return list(self.collection.find(
            {
                'plot': {"$exists": True},
                'plot_embedding_langchain': {"$exists": False}
            },
            {'_id': 1, 'plot': 1}
        ).limit(batch_size))

    def process_batch(self, documents: List[Dict[str, Any]]) -> tuple:
        """Process a batch of documents and return embeddings with their IDs."""
        embedding_matrix = []
        doc_ids = []
        failed_docs = []

        for doc in documents:
            try:
                embedding = self.embeddings.embed_query(doc['plot'])
                embedding_matrix.append(embedding)
                doc_ids.append(doc['_id'])
            except Exception as e:
                logger.error(f"Failed to process document {doc['_id']}: {str(e)}")
                failed_docs.append(doc['_id'])
                continue

        return np.array(embedding_matrix), doc_ids, failed_docs

    def reduce_dimensions(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """Reduce embedding dimensions using PCA."""
        n_samples = len(embedding_matrix)
        
        if n_samples < self.target_dim:
            padding = np.zeros((self.target_dim - n_samples, embedding_matrix.shape[1]))
            embedding_matrix = np.vstack([embedding_matrix, padding])

        pca = PCA(n_components=self.target_dim)
        reduced = pca.fit_transform(embedding_matrix)
        return reduced[:n_samples]  # Return only the original samples

    def update_mongodb(self, doc_ids: List[str], embeddings: np.ndarray) -> tuple:
        """Update documents with their embeddings."""
        success_count = 0
        failed_updates = []

        for doc_id, embedding in zip(doc_ids, embeddings):
            try:
                result = self.collection.update_one(
                    {'_id': doc_id},
                    {'$set': {'plot_embedding_langchain': embedding.tolist()}}
                )
                if result.modified_count > 0:
                    success_count += 1
                else:
                    failed_updates.append(doc_id)
            except Exception as e:
                logger.error(f"Failed to update document {doc_id}: {str(e)}")
                failed_updates.append(doc_id)

        return success_count, failed_updates

    def process_all_documents(self, max_documents: int = None) -> Dict[str, Any]:
        """Process all unprocessed documents in batches."""
        start_time = time.time()
        total_processed = 0
        total_failed = 0
        total_updates = 0
        failed_docs = []
        failed_updates = []

        while True:
            # Check if we've reached the maximum number of documents
            if max_documents and total_processed >= max_documents:
                break

            # Calculate remaining documents to process
            remaining = max_documents - total_processed if max_documents else self.batch_size
            current_batch_size = min(self.batch_size, remaining) if max_documents else self.batch_size

            # Fetch batch
            documents = self.get_unprocessed_documents(current_batch_size)
            if not documents:
                break

            logger.info(f"Processing batch of {len(documents)} documents...")

            # Process batch
            embedding_matrix, doc_ids, batch_failed_docs = self.process_batch(documents)
            failed_docs.extend(batch_failed_docs)

            if len(embedding_matrix) > 0:
                # Reduce dimensions
                reduced_embeddings = self.reduce_dimensions(embedding_matrix)

                # Update MongoDB
                success_count, batch_failed_updates = self.update_mongodb(doc_ids, reduced_embeddings)
                failed_updates.extend(batch_failed_updates)

                total_updates += success_count
                total_failed += len(batch_failed_docs) + len(batch_failed_updates)
            
            total_processed += len(documents)

            # Log progress
            logger.info(f"Processed: {total_processed}, "
                       f"Successful updates: {total_updates}, "
                       f"Failed: {total_failed}")

        end_time = time.time()
        processing_time = end_time - start_time

        return {
            "total_processed": total_processed,
            "successful_updates": total_updates,
            "total_failed": total_failed,
            "failed_documents": failed_docs,
            "failed_updates": failed_updates,
            "processing_time": processing_time
        }

    def get_stats(self) -> Dict[str, int]:
        """Get current statistics about processed and unprocessed documents."""
        total_docs = self.collection.count_documents({'plot': {"$exists": True}})
        processed_docs = self.collection.count_documents({'plot_embedding_langchain': {"$exists": True}})
        unprocessed_docs = self.collection.count_documents({
            'plot': {"$exists": True},
            'plot_embedding_langchain': {"$exists": False}
        })

        return {
            "total_documents": total_docs,
            "processed_documents": processed_docs,
            "unprocessed_documents": unprocessed_docs
        }

def main():
    # Initialize processor
    processor = BatchEmbeddingProcessor(
        batch_size=50,  # Process 50 documents at a time
        target_dim=2048
    )

    # Get initial stats
    initial_stats = processor.get_stats()
    logger.info("Initial statistics:")
    logger.info(f"Total documents: {initial_stats['total_documents']}")
    logger.info(f"Previously processed: {initial_stats['processed_documents']}")
    logger.info(f"Remaining to process: {initial_stats['unprocessed_documents']}")

    # Process documents
    results = processor.process_all_documents(max_documents=1500)

    # Log final results
    logger.info("\nProcessing completed!")
    logger.info(f"Total documents processed: {results['total_processed']}")
    logger.info(f"Successful updates: {results['successful_updates']}")
    logger.info(f"Failed operations: {results['total_failed']}")
    logger.info(f"Processing time: {results['processing_time']:.2f} seconds")

    # Get final stats
    final_stats = processor.get_stats()
    logger.info("\nFinal statistics:")
    logger.info(f"Total documents with embeddings: {final_stats['processed_documents']}")
    logger.info(f"Remaining unprocessed: {final_stats['unprocessed_documents']}")

if __name__ == "__main__":
    main()