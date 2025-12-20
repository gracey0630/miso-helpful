import pandas as pd
import json
import os
from pathlib import Path

def process_flavor_network(raw_data_path, output_path):
    """
    Processes the backbone.csv flavor network and srep00196-s3.csv cuisine data.
    Corresponds to logic in flavor_network_extraction.ipynb.
    """
    raw_data_path = Path(raw_data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Processing flavor data from {raw_data_path}...")

    # --- Part 1: Ingredient Pairs (backbone.csv) ---
    backbone_file = raw_data_path / "backbone.csv"
    if backbone_file.exists():
        df = pd.read_csv(backbone_file)
        
        # Set headers as defined in notebook
        df.columns = ["ingredient1", "ingredient2", "num_shared_compound", "category", "prevalence"]
        
        data = {
            "ingredient_pairs": [],
            "ingredients": {}
        }
        
        processed_pairs = set()

        for _, row in df.iterrows():
            # Handle ingredient pairs (sort to avoid duplicates)
            pair = tuple(sorted([row["ingredient1"], row["ingredient2"]]))
            
            if pair not in processed_pairs:
                data["ingredient_pairs"].append({
                    "ingredient1": pair[0],
                    "ingredient2": pair[1],
                    "num_shared_compound": int(row["num_shared_compound"])
                })
                processed_pairs.add(pair)
            
            # Handle ingredient info
            if row["ingredient1"] not in data["ingredients"]:
                data["ingredients"][row["ingredient1"]] = {
                    "category": row["category"],
                    "prevalence": float(row["prevalence"])
                }

        # Save ingredient_data.json
        with open(output_path / "ingredient_data.json", "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("✓ Created ingredient_data.json")
    else:
        print("x backbone.csv not found.")

    # --- Part 2: Cuisine Data (srep00196-s3.csv) ---
    cuisine_file = raw_data_path / "srep00196-s3.csv"
    if cuisine_file.exists():
        conn = {}
        with open(cuisine_file, "r") as file:
            for line in file:
                s = line.strip().split(",")
                cuisine = s[0]
                ingredients = s[1:]
                
                if cuisine not in conn:
                    conn[cuisine] = [ingredients]
                else:
                    conn[cuisine].append(ingredients)

        # Save cuisine_ingredient_data.json
        with open(output_path / "cuisine_ingredient_data.json", "w", encoding='utf-8') as file:
            json.dump(conn, file, indent=2, ensure_ascii=False)
        print("✓ Created cuisine_ingredient_data.json")
    else:
        print("x srep00196-s3.csv not found.")

if __name__ == "__main__":
    # Default paths assuming running from project root
    process_flavor_network("data/raw", "data/processed")