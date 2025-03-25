from pymongo import MongoClient
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from PyPDF2 import PdfReader
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
import logging
import os
import time
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pdf_processing.log"),
    ],
)
logger = logging.getLogger(__name__)

# MongoDB setup
def get_mongodb_client(uri: str):
    try:
        client = MongoClient(uri)
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


class PDFVectorProcessor:
    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        collection_name: str,
        embedding_model: str = "llama3.2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 10,
        target_dim: int = 2048,
    ):
        # MongoDB connection
        self.client = get_mongodb_client(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

        # Processing parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.target_dim = target_dim

        # Initialize embeddings
        try:
            self.embeddings = OllamaEmbeddings(model=embedding_model)
            logger.info(f"Successfully initialized {embedding_model} embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings model: {e}")
            raise

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            reader = PdfReader(pdf_path)
            text = []
            for i, page in enumerate(tqdm(reader.pages, desc="Extracting PDF pages")):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {i}: {e}")
                    continue

            full_text = "\n".join(text)
            logger.info(f"Successfully extracted {len(reader.pages)} pages from PDF")
            return full_text

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise

    def process_batch(self, documents: List[Any]) -> np.ndarray:
        embeddings_list = []
        for doc in documents:
            try:
                # Convert list to NumPy array explicitly
                embedding = np.array(self.embeddings.embed_query(doc.page_content))

                # Reduce dimensions if necessary
                if embedding.size > self.target_dim:
                    embedding = self.reduce_dimensions(embedding.reshape(1, -1))[0]

                embeddings_list.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")
                embeddings_list.append(np.zeros(self.target_dim))

        # Convert list of embeddings to a NumPy array
        return np.array(embeddings_list)

    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        try:
            # Determine the appropriate number of components
            n_components = min(embeddings.shape[0], embeddings.shape[1], self.target_dim)

            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(embeddings)
            logger.info(f"Successfully reduced dimensions to {n_components}")
            return reduced
        except Exception as e:
            logger.error(f"PCA reduction failed: {e}")
            return embeddings  # Return original embeddings if reduction fails

    def store_vectors_in_chunks(self, documents: List[Any]) -> None:
        """
        Stores documents with embeddings in MongoDB in chunks.
        """
        try:
            for i in tqdm(
                range(0, len(documents), self.batch_size),
                desc="Uploading to MongoDB in chunks",
            ):
                batch_documents = documents[i : i + self.batch_size]
                MongoDBAtlasVectorSearch.from_documents(
                    documents=[
                        {
                            "content": doc.page_content,
                            "metadata": {**doc.metadata, "embedding": doc.metadata["embedding"]},
                        }
                        for doc in batch_documents
                    ],
                    embedding=self.embeddings,
                    collection=self.collection,
                )
            logger.info(f"Successfully stored {len(documents)} documents in MongoDB")
        except Exception as e:
            logger.error(f"Failed to store vectors in MongoDB: {e}")
            raise

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Processes the PDF file, generates embeddings, reduces dimensions to 2048,
        and uploads the chunks to MongoDB.
        """
        start_time = time.time()
        stats = {
            "total_pages": 0,
            "total_chunks": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "processing_time": 0,
        }

        try:
            # Step 1: Extract text from the PDF
            pdf_text = self.extract_text_from_pdf(pdf_path)

            # Step 2: Split text into chunks
            documents = self.text_splitter.create_documents([pdf_text])
            stats["total_chunks"] = len(documents)
            logger.info(f"Created {len(documents)} text chunks")

            # Step 3: Generate embeddings for each chunk and reduce dimensions
            processed_documents = []
            for i in tqdm(range(0, len(documents), self.batch_size), desc="Processing batches"):
                batch = documents[i : i + self.batch_size]
                batch_embeddings = self.process_batch(batch)

                for doc, embedding in zip(batch, batch_embeddings):
                    try:
                        # Store the embedding in document metadata
                        doc.metadata["embedding"] = embedding.tolist()
                        processed_documents.append(doc)
                    except Exception as e:
                        logger.warning(f"Failed to process document: {e}")

            # Step 4: Upload processed chunks to MongoDB
            self.store_vectors_in_chunks(processed_documents)

            # Update stats
            stats["successful_embeddings"] = len(processed_documents)
            stats["processing_time"] = time.time() - start_time

            logger.info("PDF processing completed successfully")
            return stats

        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise


def main():
    config = {
        "mongo_uri": "your_mongodb_connection_string",
        "db_name": "Harry_Potter",
        "collection_name": "collection_of_harry_potter_text_blobs",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "batch_size": 10,
    }

    try:
        processor = PDFVectorProcessor(
            mongo_uri=config["mongo_uri"],
            db_name=config["db_name"],
            collection_name=config["collection_name"],
            embedding_model="llama3.2",
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            batch_size=config["batch_size"],
            target_dim=2048,
        )

        pdf_path = "./sample_files/harrypotter.pdf"
        stats = processor.process_pdf(pdf_path)

        logger.info("\nProcessing Summary:")
        logger.info(f"Total chunks processed: {stats['total_chunks']}")
        logger.info(f"Successful embeddings: {stats['successful_embeddings']}")
        logger.info(f"Failed embeddings: {stats.get('failed_embeddings', 0)}")
        logger.info(f"Total processing time: {stats['processing_time']:.2f} seconds")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
