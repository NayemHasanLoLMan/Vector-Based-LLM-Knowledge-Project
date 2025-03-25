# from pymongo import MongoClient
# from langchain_ollama import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import MongoDBAtlasVectorSearch
# from PyPDF2 import PdfReader
# from sklearn.decomposition import PCA
# from tqdm import tqdm
# import numpy as np
# import key_param

# # MongoDB setup
# client = MongoClient(key_param.MONGO_URI)
# db_name = "Harry_Potter"
# collection_name = "collection_of_harry_potter_text_blobs"
# collection = client[db_name][collection_name]

# # Step 1: Extract text from the PDF
# def extract_text_from_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     text = ""
#     for page in reader.pages:
#         if page.extract_text():  # Ensure text is extracted; handle pages with no text
#             text += page.extract_text()
#             text += "\n"
#     return text

# pdf_path = './sample_files/harrypotter.pdf'
# pdf_text = extract_text_from_pdf(pdf_path)

# # Step 2: Split the extracted text into smaller chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# split_documents = text_splitter.create_documents([pdf_text])

# # Step 3: Generate embeddings for all text chunks with progress tracking
# embeddings = OllamaEmbeddings(model="llama3.2")
# original_embeddings = []
# print("Generating embeddings...")
# for document in tqdm(split_documents, desc="Embedding generation", unit="chunk"):
#     try:
#         embedding = embeddings.embed_query(document.page_content)
#         original_embeddings.append(embedding)
#     except Exception as e:
#         print(f"Error generating embedding for document: {e}")
#         original_embeddings.append(np.zeros(2048))  # Fallback to a zero vector if embedding fails

# # Step 4: Apply PCA for dimensionality reduction
# print("Reducing dimensions with PCA...")
# embedding_matrix = np.array(original_embeddings)
# pca = PCA(n_components=min(2048, embedding_matrix.shape[1]))  # Adjust components dynamically
# reduced_embeddings = pca.fit_transform(embedding_matrix)

# # Assign reduced embeddings back to documents
# for document, reduced_embedding in zip(split_documents, reduced_embeddings):
#     document.embedding = reduced_embedding.tolist()  # Convert to list for MongoDB compatibility

# # Step 5: Store the chunked documents and their embeddings in MongoDB
# print("Storing embeddings in MongoDB...")
# try:
#     vector_store = MongoDBAtlasVectorSearch.from_documents(
#         documents=split_documents,
#         embedding=embeddings,
#         collection=collection
#     )
#     print(f"Successfully stored {len(split_documents)} chunks in MongoDB!")
# except Exception as e:
#     print(f"Error storing documents in MongoDB: {e}")


############################################################################################################


# from pymongo import MongoClient
# from langchain_ollama import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import MongoDBAtlasVectorSearch
# from PyPDF2 import PdfReader
# from sklearn.decomposition import PCA
# from tqdm import tqdm
# import numpy as np
# import key_param

# # MongoDB setup
# client = MongoClient(key_param.MONGO_URI)
# db_name = "Harry_Potter"
# collection_name = "collection_of_harry_potter_text_blobs"
# collection = client[db_name][collection_name]

# # Step 1: Extract text from the PDF
# def extract_text_from_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     text = ""
#     for page in reader.pages:
#         if page.extract_text():  # Ensure text is extracted; handle pages with no text
#             text += page.extract_text()
#             text += "\n"
#     return text

# pdf_path = './sample_files/harrypotter.pdf'
# pdf_text = extract_text_from_pdf(pdf_path)

# # Step 2: Split the extracted text into smaller chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# split_documents = text_splitter.create_documents([pdf_text])

# # Step 3: Generate embeddings for all text chunks with progress tracking
# embeddings = OllamaEmbeddings(model="llama3.2")
# original_embeddings = []
# print("Generating embeddings...")
# for document in tqdm(split_documents, desc="Embedding generation", unit="chunk"):
#     try:
#         embedding = embeddings.embed_query(document.page_content)
#         if len(embedding) != 3072:  # Validate embedding size
#             raise ValueError(f"Unexpected embedding size: {len(embedding)}")
#         original_embeddings.append(embedding)
#     except Exception as e:
#         print(f"Error generating embedding for document: {e}")
#         original_embeddings.append(np.zeros(3072))  # Fallback to a zero vector if embedding fails

# # Step 4: Apply PCA for dimensionality reduction (3072 -> 2048)
# print("Reducing dimensions with PCA...")
# embedding_matrix = np.array(original_embeddings)
# pca = PCA(n_components=2048)  # Reduce to exactly 2048 dimensions
# reduced_embeddings = pca.fit_transform(embedding_matrix)

# # Assign reduced embeddings back to documents
# for document, reduced_embedding in zip(split_documents, reduced_embeddings):
#     document.embedding = reduced_embedding.tolist()  # Convert to list for MongoDB compatibility

# # Step 5: Store the chunked documents and their embeddings in MongoDB
# print("Storing embeddings in MongoDB...")
# try:
#     vector_store = MongoDBAtlasVectorSearch.from_documents(
#         documents=split_documents,
#         embedding=embeddings,
#         collection=collection
#     )
#     print(f"Successfully stored {len(split_documents)} chunks in MongoDB!")
# except Exception as e:
#     print(f"Error storing documents in MongoDB: {e}")



############################################################################################################


# from pymongo import MongoClient
# from langchain_ollama import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import MongoDBAtlasVectorSearch
# from PyPDF2 import PdfReader
# from sklearn.decomposition import PCA
# from tqdm import tqdm
# import numpy as np
# import logging
# import os
# from typing import List, Dict, Any, Optional
# import time

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler('pdf_processing.log')
#     ]
# )
# logger = logging.getLogger(__name__)

# # MongoDB setup
# client = MongoClient("mongodb+srv://hasanmahmudnayeem3027:JT8J4TYqlCLTVgJu@cluster0.qthig.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
# db_name = "Harry_Potter"
# collection_name = "collection_of_harry_potter_text_blobs"
# collection = client[db_name][collection_name]

# class PDFVectorProcessor:
#     def __init__(
#         self,
#         embedding_model: str = "llama3.2",
#         chunk_size: int = 1000,
#         chunk_overlap: int = 100,
#         batch_size: int = 10,
#         target_dim: int = 2048
#     ):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.batch_size = batch_size
#         self.target_dim = target_dim

#         try:
#             self.embeddings = OllamaEmbeddings(model=embedding_model)
#             logger.info(f"Successfully initialized {embedding_model} embeddings")
#         except Exception as e:
#             logger.error(f"Failed to initialize embeddings model: {e}")
#             raise

#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap
#         )

#     def extract_text_from_pdf(self, pdf_path: str) -> str:
#         if not os.path.exists(pdf_path):
#             raise FileNotFoundError(f"PDF file not found: {pdf_path}")

#         try:
#             reader = PdfReader(pdf_path)
#             text = []
#             for i, page in enumerate(tqdm(reader.pages, desc="Extracting PDF pages")):
#                 try:
#                     page_text = page.extract_text()
#                     if page_text:
#                         text.append(page_text)
#                 except Exception as e:
#                     logger.warning(f"Failed to extract text from page {i}: {e}")
#                     continue

#             full_text = "\n".join(text)
#             logger.info(f"Successfully extracted {len(reader.pages)} pages from PDF")
#             return full_text

#         except Exception as e:
#             logger.error(f"Error processing PDF {pdf_path}: {e}")
#             raise

#     def process_batch(self, documents: List[Any]) -> List[np.ndarray]:
#         embeddings_list = []
#         for doc in documents:
#             try:
#                 embedding = self.embeddings.embed_query(doc.page_content)
#                 embeddings_list.append(embedding)
#             except Exception as e:
#                 logger.warning(f"Failed to generate embedding: {e}")
#                 embeddings_list.append(np.zeros(self.target_dim))
#         return embeddings_list

#     def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
#         try:
#             pca = PCA(n_components=self.target_dim)
#             reduced = pca.fit_transform(embeddings)
#             logger.info(f"Successfully reduced dimensions to {self.target_dim}")
#             return reduced
#         except Exception as e:
#             logger.error(f"PCA reduction failed: {e}")
#             raise

#     def rewrite_existing_embeddings(self) -> None:
#         cursor = self.collection.find({"plot_embedding_langchain": {"$exists": True}})
#         updated_count = 0

#         for doc in tqdm(cursor, desc="Rewriting embeddings"):
#             original_embedding = np.array(doc["plot_embedding_langchain"])
#             if original_embedding.shape[0] == 3072:
#                 reduced_embedding = self.reduce_dimensions(original_embedding.reshape(1, -1))
#                 self.collection.update_one(
#                     {"_id": doc["_id"]},
#                     {"$set": {"plot_embedding_langchain": reduced_embedding[0].tolist()}}
#                 )
#                 updated_count += 1

#         logger.info(f"Successfully updated {updated_count} embeddings to {self.target_dim} dimensions")

#     def store_vectors(self, documents: List[Any]) -> None:
#         try:
#             vector_store = MongoDBAtlasVectorSearch.from_documents(
#                 documents=documents,
#                 embedding=self.embeddings,
#                 collection=self.collection
#             )
#             logger.info(f"Successfully stored {len(documents)} documents in MongoDB")
#         except Exception as e:
#             logger.error(f"Failed to store vectors in MongoDB: {e}")
#             raise

#     def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
#         start_time = time.time()
#         stats = {
#             "total_pages": 0,
#             "total_chunks": 0,
#             "successful_embeddings": 0,
#             "failed_embeddings": 0,
#             "processing_time": 0
#         }

#         try:
#             pdf_text = self.extract_text_from_pdf(pdf_path)

#             documents = self.text_splitter.create_documents([pdf_text])
#             stats["total_chunks"] = len(documents)
#             logger.info(f"Created {len(documents)} text chunks")

#             all_embeddings = []
#             for i in tqdm(range(0, len(documents), self.batch_size), desc="Processing batches"):
#                 batch = documents[i:i + self.batch_size]
#                 batch_embeddings = self.process_batch(batch)
#                 all_embeddings.extend(batch_embeddings)
#                 stats["successful_embeddings"] += len(batch_embeddings)

#             reduced_embeddings = self.reduce_dimensions(np.array(all_embeddings))

#             for doc, embedding in zip(documents, reduced_embeddings):
#                 doc.embedding = embedding.tolist()

#             self.store_vectors(documents)

#             stats["processing_time"] = time.time() - start_time
#             logger.info("PDF processing completed successfully")
#             return stats

#         except Exception as e:
#             logger.error(f"PDF processing failed: {e}")
#             raise

# def main():
#     config = {
#         "mongo_uri": "mongodb+srv://hasanmahmudnayeem3027:JT8J4TYqlCLTVgJu@cluster0.qthig.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
#         "db_name": "Harry_Potter",
#         "collection_name": "collection_of_harry_potter_text_blobs",
#         "chunk_size": 1000,
#         "chunk_overlap": 100,
#         "batch_size": 10
#     }

#     try:
#         processor = PDFVectorProcessor(
#             embedding_model="llama3.2",
#             chunk_size=config["chunk_size"],
#             chunk_overlap=config["chunk_overlap"],
#             batch_size=config["batch_size"],
#             target_dim=2048
#         )

#         processor.rewrite_existing_embeddings()

#         pdf_path = './sample_files/harrypotter.pdf'
#         stats = processor.process_pdf(pdf_path)

#         logger.info("\nProcessing Summary:")
#         logger.info(f"Total chunks processed: {stats['total_chunks']}")
#         logger.info(f"Successful embeddings: {stats['successful_embeddings']}")
#         logger.info(f"Failed embeddings: {stats['failed_embeddings']}")
#         logger.info(f"Total processing time: {stats['processing_time']:.2f} seconds")

#     except Exception as e:
#         logger.error(f"Processing failed: {e}")
#         raise

# if __name__ == "__main__":
#     main()

############################################################################################################



from pymongo import MongoClient
from urllib.parse import quote_plus
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from PyPDF2 import PdfReader
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
import logging
import os
from typing import List, Dict, Any, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pdf_processing.log')
    ]
)
logger = logging.getLogger(__name__)

class PDFVectorProcessor:
    def __init__(
        self,
        username: str,
        password: str,
        cluster_url: str,
        db_name: str,
        collection_name: str,
        embedding_model: str = "llama3.2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 10,
        target_dim: int = 2048
    ):
        # Secure MongoDB connection
        try:
            username = quote_plus(username)
            password = quote_plus(password)
            mongo_uri = f"mongodb+srv://{username}:{password}@{cluster_url}/?retryWrites=true&w=majority&authSource=admin"
            
            self.client = MongoClient(
                mongo_uri, 
                username=username, 
                password=password,
                authSource="admin"
            )
            self.collection = self.client[db_name][collection_name]
            logger.info(f"Connected to MongoDB database: {db_name}.{collection_name}")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise

        # Embedding and text processing configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.target_dim = target_dim

        try:
            self.embeddings = OllamaEmbeddings(model=embedding_model)
            logger.info(f"Successfully initialized {embedding_model} embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings model: {e}")
            raise

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
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

    def process_batch(self, documents: List[Any]) -> List[np.ndarray]:
        embeddings_list = []
        for doc in documents:
            try:
                embedding = self.embeddings.embed_query(doc.page_content)
                # Ensure embedding is a numpy array
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                embeddings_list.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")
                embeddings_list.append(np.zeros(self.target_dim))
        return embeddings_list

    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        try:
            pca = PCA(n_components=self.target_dim)
            reduced = pca.fit_transform(embeddings)
            logger.info(f"Successfully reduced dimensions to {self.target_dim}")
            return reduced
        except Exception as e:
            logger.error(f"PCA reduction failed: {e}")
            raise

    def rewrite_existing_embeddings(self) -> None:
        cursor = self.collection.find({"plot_embedding_langchain": {"$exists": True}})
        updated_count = 0

        for doc in tqdm(cursor, desc="Rewriting embeddings"):
            original_embedding = np.array(doc["plot_embedding_langchain"])
            if original_embedding.shape[0] == 3072:
                reduced_embedding = self.reduce_dimensions(original_embedding.reshape(1, -1))
                self.collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"plot_embedding_langchain": reduced_embedding[0].tolist()}}
                )
                updated_count += 1

        logger.info(f"Successfully updated {updated_count} embeddings to {self.target_dim} dimensions")

    def store_vectors(self, documents: List[Any]) -> None:
        try:
            vector_store = MongoDBAtlasVectorSearch.from_documents(
                documents=[
                    {
                        "content": doc.page_content,
                        "metadata": {**doc.metadata, "embedding": doc.metadata["embedding"]}
                    } for doc in documents
                ],
                embedding=self.embeddings,
                collection=self.collection
            )
            logger.info(f"Successfully stored {len(documents)} documents in MongoDB")
        except Exception as e:
            logger.error(f"Failed to store vectors in MongoDB: {e}")
            raise

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        start_time = time.time()
        stats = {
            "total_pages": 0,
            "total_chunks": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "processing_time": 0
        }

        try:
            pdf_text = self.extract_text_from_pdf(pdf_path)
            documents = self.text_splitter.create_documents([pdf_text])
            stats["total_chunks"] = len(documents)
            logger.info(f"Created {len(documents)} text chunks")

            # Process and store vectors in smaller sections
            for i in tqdm(range(0, len(documents), self.batch_size), desc="Processing and storing sections"):
                section_documents = documents[i:i + self.batch_size]
                
                # Generate embeddings for the current section
                section_embeddings = self.process_batch(section_documents)
                
                # Prepare processed documents with embeddings
                processed_section_docs = []
                for doc, embedding in zip(section_documents, section_embeddings):
                    # Convert embedding to list if it's a numpy array
                    if isinstance(embedding, np.ndarray):
                        if embedding.size == 3072:
                            embedding = self.reduce_dimensions(embedding.reshape(1, -1))[0]
                        embedding = embedding.tolist()
                    
                    doc.metadata["embedding"] = embedding
                    processed_section_docs.append(doc)
                
                # Store current section's vectors
                self.store_vectors(processed_section_docs)
                
                stats["successful_embeddings"] += len(processed_section_docs)

            stats["processing_time"] = time.time() - start_time
            logger.info("PDF processing completed successfully")
            return stats

        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise

def main():
    config = {
        "username": "hasanmahmudnayeem3027",
        "password": "JT8J4TYqlCLTVgJu",
        "cluster_url": "cluster0.qthig.mongodb.net",
        "db_name": "Harry_Potter",
        "collection_name": "collection_of_harry_potter_text_blobs",
        "embedding_model": "llama3.2",
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "batch_size": 10,
        "target_dim": 2048
    }

    try:
        processor = PDFVectorProcessor(
            username=config["username"],
            password=config["password"],
            cluster_url=config["cluster_url"],
            db_name=config["db_name"],
            collection_name=config["collection_name"],
            embedding_model=config["embedding_model"],
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            batch_size=config["batch_size"],
            target_dim=config["target_dim"]
        )

        # Optional: Rewrite existing embeddings
        processor.rewrite_existing_embeddings()

        # Process PDF
        pdf_path = './sample_files/harrypotter.pdf'
        stats = processor.process_pdf(pdf_path)

        # Log processing summary
        logger.info("\nProcessing Summary:")
        logger.info(f"Total chunks processed: {stats['total_chunks']}")
        logger.info(f"Successful embeddings: {stats['successful_embeddings']}")
        logger.info(f"Total processing time: {stats['processing_time']:.2f} seconds")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()