from pymongo import MongoClient
from langchain_ollama import OllamaEmbeddings
from typing import List, Dict, Any
import gradio as gr
import logging
import re

class SemanticSearchEngine:
    def __init__(
        self, 
        mongodb_uri: str = "mongodb+srv://hasanmahmudnayeem3027:JT8J4TYqlCLTVgJu@cluster0.qthig.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
        database_name: str = "sample_mflix",
        collection_name: str = "movies",
        embedding_model: str = "llama3.2",
        vector_search_index: str = "PlotSemanticSearch",
        embedding_field: str = "plot_embedding_langchain"
    ):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        try:
            self.client = MongoClient(mongodb_uri)
            self.db = self.client[database_name]
            self.collection = self.db[collection_name]
            self.embeddings = OllamaEmbeddings(model=embedding_model)
            
            self.vector_search_index = vector_search_index
            self.embedding_field = embedding_field
        
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        
        keywords = text.split()
        if len(keywords) < 3:
            text = f"Movies about {text} theme or concept"
        
        return text

    def generate_embedding(self, text: str, target_dim: int = 2048) -> List[float]:
        try:
            processed_text = self.preprocess_text(text)
            embedding = self.embeddings.embed_query(processed_text)
            
            if len(embedding) > target_dim:
                return embedding[:target_dim]
            return embedding + [embedding[-1]] * (target_dim - len(embedding))
        
        except Exception as e:
            self.logger.error(f"Embedding error: {e}")
            return [0.0] * target_dim

    def search(
        self,
        query: str,
        target_dim: int = 2048,
        top_k: int = 10,
        num_candidates: int = 1000
    ) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.generate_embedding(query, target_dim)

            pipeline = [
                {
                    "$vectorSearch": {
                        "queryVector": query_embedding,
                        "path": self.embedding_field,
                        "numCandidates": num_candidates,
                        "limit": top_k,
                        "index": self.vector_search_index,
                    }
                },
                {
                    "$project": {
                        "title": 1,
                        "plot": 1,
                        "genres": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {"$sort": {"score": -1}},
                {"$limit": top_k}
            ]

            return list(self.collection.aggregate(pipeline))
        
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return []

def format_results(results):
    if not results:
        return "<p>No results found.</p>"

    output = "<h3>Search Results</h3><ol>"
    for doc in results:
        title = doc.get("title", "Unknown Title")
        plot = doc.get("plot", "No plot available")
        genres = doc.get("genres", [])
        score = doc.get("score", 0)

        output += f"""
            <li>
                <strong>{title}</strong><br>
                <small>Genres: {", ".join(genres)}</small><br>
                <small>Relevance Score: {score:.2f}</small><br>
                {plot}
            </li>
        """
    output += "</ol>"
    return output

def create_search_interface():
    search_engine = SemanticSearchEngine()

    def handle_search(query, top_k):
        if not query.strip():
            return "<p>Please enter a search query.</p>"
        
        results = search_engine.search(query, top_k=top_k)
        return format_results(results)

    interface = gr.Blocks(title="Semantic Movie Search")
    
    with interface:
        gr.Markdown("# 🎬 Semantic Movie Search")
        
        with gr.Row():
            query_input = gr.Textbox(
                label="Search Query", 
                placeholder="Enter movie description...",
                lines=2
            )
            top_k = gr.Slider(
                minimum=1, 
                maximum=20, 
                value=5, 
                step=1, 
                label="Number of Results"
            )
        
        search_button = gr.Button("Search", variant="primary")
        results_output = gr.HTML(label="Search Results")

        search_button.click(
            fn=handle_search, 
            inputs=[query_input, top_k], 
            outputs=[results_output]
        )

    return interface

def main():
    interface = create_search_interface()
    interface.launch(share=True)

if __name__ == "__main__":
    main()
