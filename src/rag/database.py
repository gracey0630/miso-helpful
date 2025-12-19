import os
import json
import glob
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pathlib import Path
from .chunking import (
    chunk_cuisine_ingredients_dict, chunk_fcs_json, chunk_recipe_json, chunk_ingredient_data_json, 
    chunk_csv_simple, chunk_reddit_json, chunk_3a2m_recipe_json, apply_recursive_chunking
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
                chunks = chunker(data, filename)
                all_chunks.extend(chunks)
                print(f"✓ Loaded {len(chunks)} chunks from {filename}")

        # 2a. Recipes (foc folder)
        foc_path = os.path.join(processed_data_path, "foc")
        if os.path.exists(foc_path):
            for file_path in glob.glob(os.path.join(foc_path, "*.json")):
                if "sections" in file_path: continue # handled separately if needed
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                chunks = chunk_recipe_json(data, os.path.basename(file_path))
                all_chunks.extend(chunks)
                print(f"✓ Loaded {len(chunks)} chunks from {file_path}")

        # 2b. 3A2M Recipes
        recipes_3a2m_path = os.path.join(processed_data_path, "3a2m_recipe_data.json")
        if os.path.exists(recipes_3a2m_path):
            try:
                with open(recipes_3a2m_path, 'r', encoding='utf-8') as f:
                    recipe_array = json.load(f)  # Load array of recipes
                
                recipe_count = 0
                for recipe_data in recipe_array:
                    chunks = chunk_3a2m_recipe_json(recipe_data, "3a2m_recipe_data.json")
                    all_chunks.extend(chunks)
                    recipe_count += 1
                
                print(f"✓ Loaded {recipe_count} 3A2M recipes")
            except Exception as e:
                print(f"✗ Error loading 3A2M recipes: {e}")

        # 3. Ingredient Pairings
        ing_path = os.path.join(processed_data_path, "ingredient_data.json")
        if os.path.exists(ing_path):
            with open(ing_path, 'r') as f:
                data = json.load(f)
            chunks = chunk_ingredient_data_json(data, "ingredient_data.json")
            all_chunks.extend(chunks)
        print(f"✓ Loaded {len(chunks)} chunks from ingredient_data.json")

        # 4. CSVs (Cooking Methods)
        for csv_path in glob.glob(os.path.join(processed_data_path, "cooking_methods.csv")):
            chunks = chunk_csv_simple(csv_path)
            all_chunks.extend(chunks)
        print(f"✓ Loaded {len(chunks)} chunks from cooking_methods.csv")


        # 5. Reddit Data
        reddit_chunks = []
        reddit_path = os.path.join(processed_data_path, "reddit")
        if os.path.exists(reddit_path):
            for json_file in glob.glob(os.path.join(reddit_path, "*.json")):
                chunks = chunk_reddit_json(json_file)
                reddit_chunks.extend(chunks)
                print(f"✓ Loaded {len(chunks)} chunks from {json_file}")
        # Apply recursive chunking for long texts
        reddit_chunks = apply_recursive_chunking(reddit_chunks)
        print(f"✓ Loaded total of {len(reddit_chunks)} chunks from Reddit, after recursive chunking")
        all_chunks.extend(reddit_chunks)

        # 6. Cuisine Ingredient Data
        cuisine_path = os.path.join(processed_data_path, "cuisine_ingredient_data.json")
        if os.path.exists(cuisine_path):
            with open(cuisine_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            chunks = chunk_cuisine_ingredients_dict(data, "cuisine_ingredient_data.json")
            all_chunks.extend(chunks)
        print(f"✓ Loaded {len(chunks)} chunks from cuisine_ingredient_data.json")

        # 7. Batch Add to Chroma
        if all_chunks:
            print(f"Adding {len(all_chunks)} chunks to database...")
            # Chroma handles batching, but for safety with large datasets:
            batch_size = 5000
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i+batch_size]
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