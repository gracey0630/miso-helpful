# ===== CHUNKING FUNCTIONS =====

def chunk_fcs_json(data, source_file):
    chunks = []
    
    # Introduction
    chunks.append({
        'text': f"{data['title']}\n\n{data['introduction']}",
        'metadata': {'type': 'introduction', 'source': source_file}
    })
    
    # Cooking methods and techniques
    for method_name, method_data in data['cooking_methods'].items():
        if isinstance(method_data, dict) and 'techniques' in method_data:
            for technique_name, technique_desc in method_data['techniques'].items():
                chunks.append({
                    'text': f"Cooking Method: {method_name.replace('_', ' ').title()}\n\n"
                            f"{method_data['description']}\n\n"
                            f"Technique: {technique_name}\n{technique_desc}",
                    'metadata': {
                        'type': 'technique',
                        'method': method_name,
                        'technique': technique_name,
                        'source': source_file
                    }
                })
        else:
            chunks.append({
                'text': f"Cooking Method: {method_name.title()}\n\n{method_data['description']}",
                'metadata': {'type': 'method', 'method': method_name, 'source': source_file}
            })
    
    # Kitchen tools
    if 'kitchen_tools' in data:
        for category, category_data in data['kitchen_tools'].items():
            for tool_name, tool_desc in category_data['items'].items():
                chunks.append({
                    'text': f"Kitchen Tool: {tool_name}\n\n{tool_desc}",
                    'metadata': {
                        'type': 'kitchen_tool',
                        'category': category,
                        'tool': tool_name,
                        'source': source_file
                    }
                })
    
    return chunks


def chunk_healthy_cooking(data, source_file):
    chunks = []
    
    for method_name, method_desc in data['cooking_methods'].items():
        chunks.append({
            'text': f"Healthy Cooking Method: {method_name}\n\n{method_desc}",
            'metadata': {
                'type': 'healthy_method',
                'method': method_name.lower(),
                'source': source_file
            }
        })
    
    if 'food_preparation_tips' in data:
        tips_text = "Healthy Food Preparation Tips:\n\n" + "\n\n".join(
            f"• {tip}" for tip in data['food_preparation_tips']
        )
        chunks.append({
            'text': tips_text,
            'metadata': {'type': 'preparation_tips', 'source': source_file}
        })
    
    return chunks


def chunk_recipe_json(data, source_file):
    ingredients_text = "\n".join(
        f"- {ing['item']}: {ing['quantity']}" 
        for ing in data['ingredients']
    )
    
    recipe_text = f"""Recipe: {data['dish_name']}

Prep time: {data.get('prep_time', 'N/A')}
Cooking time: {data.get('cooking_time', 'N/A')}
Portions: {data.get('portions', 'N/A')}

Ingredients:
{ingredients_text}"""
    
    return [{
        'text': recipe_text,
        'metadata': {
            'type': 'recipe',
            'dish_name': data['dish_name'],
            'source': source_file
        }
    }]


def chunk_csv_simple(file_path):
    """
    Put entire CSV in one chunk (for small CSVs ~5 lines)
    """
    try:
        df = pd.read_csv(file_path)
        
        # Format as readable text
        text_parts = [f"Data from {file_path.split('/')[-1]}:\n"]
        
        for idx, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            text_parts.append(row_text)
        
        text = "\n".join(text_parts)
        
        return [{
            'text': text,
            'metadata': {
                'type': 'csv_data',
                'source': file_path,
                'filename': file_path.split('/')[-1],
                'rows': len(df),
                'columns': len(df.columns)
            }
        }]
        
    except Exception as e:
        print(f"  ✗ Error processing CSV {file_path}: {e}")
        return []

def chunk_json_simple(file_path):
    """
    Put each row of JSON in its own chunk - subject: text related to subject format
    """

    chunks = []

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        for key, val in data.items():
                chunks.append({
                    'text': f"{key}: {val}",
                    'metadata': {
                        'type': 'json_data',
                        'source': file_path,
                        'key': key,
                        'value': val
                    }
                })
                
    except Exception as e:
        print(f"  ✗ Error processing JSON {file_path}: {e}")
        return []
        
    return chunks