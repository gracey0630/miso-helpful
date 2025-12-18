import os
import json
import glob
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pathlib import Path
from .chunking import (
    chunk_cuisine_ingredients_dict, chunk_fcs_json, chunk_recipe_json, chunk_ingredient_data_json, 
    chunk_csv_simple, chunk_reddit_json, apply_recursive_chunking
)

class CookingDB:
    def __init__(self, db_path="./data/chroma_db"):
        self.db_path = db_path
        # Ensure directory exists
        Path(db_path).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = "cooking_assistant"
        
        # Use SentenceTransformer for embeddings (as in your notebook)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )

    def ingest_data(self, processed_data_path):
        """
        Loads all data from the processed folder and ingests it into ChromaDB.
        """
        print(f"Ingesting data from {processed_data_path}...")
        all_chunks = []

        # 1. FCS and Healthy Cooking (Specific JSONs)
        specific_files = {
            'FCS.json': chunk_fcs_json,
            'healthy_cooking_method.json': chunk_fcs_json # Reusing FCS logic or add specific if needed
        }
        
        for filename, chunker in specific_files.items():
            path = os.path.join(processed_data_path, filename)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                all_chunks.extend(chunker(data, filename))

        # 2. Recipes (foc folder)
        foc_path = os.path.join(processed_data_path, "foc")
        if os.path.exists(foc_path):
            for file_path in glob.glob(os.path.join(foc_path, "*.json")):
                if "sections" in file_path: continue # handled separately if needed
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                all_chunks.extend(chunk_recipe_json(data, os.path.basename(file_path)))

        # 3. Ingredient Pairings
        ing_path = os.path.join(processed_data_path, "ingredient_data.json")
        if os.path.exists(ing_path):
            with open(ing_path, 'r') as f:
                data = json.load(f)
            all_chunks.extend(chunk_ingredient_data_json(data, "ingredient_data.json"))

        # 4. CSVs (Cooking Methods)
        for csv_path in glob.glob(os.path.join(processed_data_path, "*.csv")):
            all_chunks.extend(chunk_csv_simple(csv_path))

        # 5. Reddit Data
        reddit_path = os.path.join(processed_data_path, "reddit")
        if os.path.exists(reddit_path):
            for json_file in glob.glob(os.path.join(reddit_path, "*.json")):
                all_chunks.extend(chunk_reddit_json(json_file))

        # 6. Cuisine Ingredient Data
        cuisine_path = os.path.join(processed_data_path, "cuisine_ingredient_data.json")
        if os.path.exists(cuisine_path):
            with open(cuisine_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_chunks.extend(chunk_cuisine_ingredients_dict(data, "cuisine_ingredient_data.json"))

        # 6. Apply Recursive Splitting for long chunks
        final_chunks = apply_recursive_chunking(all_chunks)

        # 7. Batch Add to Chroma
        if final_chunks:
            print(f"Adding {len(final_chunks)} chunks to database...")
            # Chroma handles batching, but for safety with large datasets:
            batch_size = 5000
            for i in range(0, len(final_chunks), batch_size):
                batch = final_chunks[i:i+batch_size]
                print(f"   -> Processing batch {i} to {i+len(batch)}...")
                self.collection.add(
                    documents=[c['text'] for c in batch],
                    metadatas=[c['metadata'] for c in batch],
                    ids=[f"doc_{k + i}" for k in range(len(batch))]
                )
            print("Ingestion complete.")
        else:
            print("No data found to ingest.")

    def query(self, text, n_results=3):
        return self.collection.query(query_texts=[text], n_results=n_results)

if __name__ == "__main__":
    # Script to run ingestion manually
    db = CookingDB()
    db.ingest_data("./data/processed")