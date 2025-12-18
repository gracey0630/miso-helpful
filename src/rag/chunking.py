import json
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_fcs_json(data, source_file):
    """
    Handles both nested (FCS.json) and flat (healthy_cooking_method.json) structures.
    """
    chunks = []
    
    # 1. Handle Introduction
    if 'introduction' in data:
        chunks.append({
            'text': f"{data.get('title', 'Introduction')}\n\n{data['introduction']}",
            'metadata': {'type': 'introduction', 'source': source_file}
        })

    # 2. Handle Cooking Methods
    if 'cooking_methods' in data:
        for method_name, method_data in data['cooking_methods'].items():
            # CASE A: Nested structure (Found in FCS.json)
            if isinstance(method_data, dict):
                description = method_data.get('description', '')
                
                # If there are sub-techniques (Poaching, Steaming, etc.)
                if 'techniques' in method_data:
                    for tech_name, tech_desc in method_data['techniques'].items():
                        chunks.append({
                            'text': f"Method: {method_name.replace('_', ' ').title()}\nCategory Description: {description}\nTechnique: {tech_name}\n{tech_desc}",
                            'metadata': {
                                'type': 'technique',
                                'method': method_name,
                                'technique': tech_name,
                                'source': source_file
                            }
                        })
                else:
                    # Dict without sub-techniques
                    chunks.append({
                        'text': f"Cooking Method: {method_name.title()}\n\n{description}",
                        'metadata': {'type': 'method', 'method': method_name, 'source': source_file}
                    })
            
            # CASE B: Flat string structure (Found in healthy_cooking_method.json)
            elif isinstance(method_data, str):
                chunks.append({
                    'text': f"Healthy Cooking Method: {method_name}\n\n{method_data}",
                    'metadata': {
                        'type': 'method', 
                        'method': method_name, 
                        'source': source_file
                    }
                })

    # 3. Handle specific arrays (like food_preparation_tips in healthy_cooking_method.json)
    if 'food_preparation_tips' in data:
        tips_text = "\n".join([f"- {tip}" for tip in data['food_preparation_tips']])
        chunks.append({
            'text': f"Healthy Food Preparation Tips:\n{tips_text}",
            'metadata': {'type': 'tips', 'source': source_file}
        })

    # 4. Handle Kitchen Tools (Found in FCS.json)
    if 'kitchen_tools' in data:
        for cat_name, cat_data in data['kitchen_tools'].items():
            if isinstance(cat_data, dict) and 'items' in cat_data:
                for tool, desc in cat_data['items'].items():
                    chunks.append({
                        'text': f"Kitchen Tool ({cat_name.title()}): {tool}\n\n{desc}",
                        'metadata': {'type': 'tool', 'tool': tool, 'source': source_file}
                    })

    return chunks

def chunk_recipe_json(data, source_file):
    # Handles foc_ingredients_*.json files
    ingredients_text = "\n".join(
        f"- {ing['item']}: {ing['quantity']}"
        for ing in data.get('ingredients', [])
    )

    recipe_text = f"""Recipe: {data.get('dish_name', 'Unknown')}\n
Prep time: {data.get('prep_time', 'N/A')}
Cooking time: {data.get('cooking_time', 'N/A')}
Portions: {data.get('portions', 'N/A')}\n
Ingredients:
{ingredients_text}"""

    return [{
        'text': recipe_text,
        'metadata': {
            'type': 'recipe',
            'dish_name': data.get('dish_name', 'Unknown'),
            'source': source_file
        }
    }]

def chunk_cuisine_ingredients_dict(cuisine_dict, source_file):
    """
    Efficient chunking: Cuisine profiles + ingredient mappings
    """
    chunks = []
    
    try:
        from collections import defaultdict
        
        # --- 1. Cuisine profiles (one per cuisine) ---
        for cuisine, recipe_list in cuisine_dict.items():
            # Count ingredient frequency
            ingredient_freq = defaultdict(int)
            for recipe in recipe_list:
                for ing in recipe:
                    ingredient_freq[ing] += 1
            
            # Get top 15 ingredients
            sorted_ingredients = sorted(ingredient_freq.items(), key=lambda x: x[1], reverse=True)
            top_ingredients = sorted_ingredients[:15]
            
            # Format as simple list
            ing_list = ', '.join([ing.replace('_', ' ').title() for ing, _ in top_ingredients])
            
            text = f"""{cuisine} Cuisine

Common ingredients: {ing_list}

Based on {len(recipe_list)} recipes with {len(ingredient_freq)} unique ingredients."""
            
            chunks.append({
                'text': text,
                'metadata': {
                    'type': 'cuisine_profile',
                    'cuisine': cuisine.lower(),
                    'num_recipes': len(recipe_list),
                    'source': source_file
                }
            })
        
        # --- 2. Ingredient-to-cuisine mappings ---
        ingredient_to_cuisines = defaultdict(set)
        
        for cuisine, recipe_list in cuisine_dict.items():
            for recipe in recipe_list:
                for ing in recipe:
                    ingredient_to_cuisines[ing].add(cuisine)
        
        # Create chunks only for ingredients used in 2+ cuisines
        for ingredient, cuisines in ingredient_to_cuisines.items():
            if len(cuisines) >= 2:
                ing_name = ingredient.replace('_', ' ').title()
                cuisine_list = ', '.join(sorted(cuisines))
                
                text = f"""Ingredient: {ing_name}

Found in {len(cuisines)} cuisines: {cuisine_list}

This ingredient appears across multiple culinary traditions."""
                
                chunks.append({
                    'text': text,
                    'metadata': {
                        'type': 'ingredient_cuisine_map',
                        'ingredient': ingredient,
                        'num_cuisines': len(cuisines),
                        'source': source_file
                    }
                })
        
        cuisine_count = len(cuisine_dict)
        ingredient_count = len([c for c in chunks if c['metadata']['type'] == 'ingredient_cuisine_map'])
        
        print(f"✓ Created {len(chunks)} chunks: {cuisine_count} cuisine profiles + {ingredient_count} ingredient mappings")
        return chunks
        
    except Exception as e:
        print(f"✗ Error processing cuisine data: {e}")
        return []

def chunk_ingredient_data_json(data, source_file):
    chunks = []
    
    # 1. Individual ingredient pairs
    if 'ingredient_pairs' in data:
        for pair in data['ingredient_pairs']:
            ing1 = pair['ingredient1'].replace('_', ' ').title()
            ing2 = pair['ingredient2'].replace('_', ' ').title()
            shared = pair['num_shared_compound']
            
            pairing_strength = "strongly" if shared > 30 else "moderately" if shared > 15 else "somewhat"
            
            text = f"Ingredient Pairing: {ing1} and {ing2}\n\n" \
                   f"These ingredients share {shared} chemical compounds, suggesting they {pairing_strength} complement each other.\n" \
                   f"Pairing strength: {pairing_strength.title()}\nShared compounds: {shared}"

            chunks.append({
                'text': text,
                'metadata': {
                    'type': 'ingredient_pairing',
                    'ingredient1': pair['ingredient1'],
                    'ingredient2': pair['ingredient2'],
                    'shared_compounds': shared,
                    'source': source_file
                }
            })

    # 2. Ingredient Info
    if 'ingredients' in data:
        for ingredient, info in data['ingredients'].items():
            ing_name = ingredient.replace('_', ' ').title()
            chunks.append({
                'text': f"Ingredient: {ing_name}\nCategory: {info.get('category', 'unknown').title()}\nPrevalence: {info.get('prevalence', 0):.2%}",
                'metadata': {
                    'type': 'ingredient_info',
                    'ingredient': ingredient,
                    'category': info.get('category'),
                    'source': source_file
                }
            })
            
    return chunks

def chunk_csv_simple(file_path):
    try:
        df = pd.read_csv(file_path)
        text_parts = [f"Data from {file_path.split('/')[-1]}:\n"]
        for _, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            text_parts.append(row_text)
        
        return [{
            'text': "\n".join(text_parts),
            'metadata': {'type': 'csv_data', 'source': file_path, 'filename': file_path.split('/')[-1]}
        }]
    except Exception as e:
        print(f"Error chunking CSV {file_path}: {e}")
        return []

def chunk_reddit_json(file_path):
    chunks = []
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        for post in data:
            # Post Chunk
            chunks.append({
                'text': f"Title: {post.get('title', '')}\n\nPost: {post.get('selftext', '')}",
                'metadata': {'type': 'reddit_post', 'source': file_path, 'post_title': post.get('title', '')}
            })
            
            # Comment Chunks (Grouped)
            comments = post.get("comments", [])
            group_size = 5
            for i in range(0, len(comments), group_size):
                group = comments[i:i+group_size]
                comment_text = f"Comments for post '{post.get('title', '')}':\n"
                for c in group:
                    comment_text += f"{c.get('author', 'User')}: {c.get('body', '')}\n"
                
                chunks.append({
                    'text': comment_text,
                    'metadata': {'type': 'reddit_comments', 'source': file_path, 'post_title': post.get('title', '')}
                })
    except Exception as e:
        print(f"Error chunking Reddit JSON {file_path}: {e}")
        return []
    return chunks

def apply_recursive_chunking(chunks, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    final_chunks = []
    for chunk in chunks:
        # Only split if necessary
        if len(chunk['text']) > chunk_size:
            text_parts = splitter.split_text(chunk['text'])
            for i, part in enumerate(text_parts):
                final_chunks.append({
                    'text': part,
                    'metadata': {**chunk['metadata'], 'chunk_part': i}
                })
        else:
            final_chunks.append(chunk)
    return final_chunks