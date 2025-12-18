import pdfplumber
import pandas as pd
import numpy as np
import re
import json
import os
from pathlib import Path

# --- Helper Functions ---

def extract_ingredients(df):
    """Parses a specific table structure used for recipes in the PDF."""
    try:
        recipe = {
            'dish_name': str(df.iloc[0, 1]).replace("\n", " "),
            'prep_time': str(df.iloc[0, 2]).replace("\n", " "),
            'cooking_time': str(df.iloc[0, 3]).replace("\n", " "),
            'portions': df.iloc[1, 1],
            'unit_size': df.iloc[1, 3],
            'ingredients': []
        }

        # Find ingredients table start
        items_row = df[df.iloc[:, 0] == 'Items'].index
        if len(items_row) > 0:
            table_start = items_row[0] + 1
            
            # Filter empty rows and reset index
            ingredients_df = df.iloc[table_start:, :].replace('', np.nan).dropna(subset=[0,2])
            ingredients_df.columns = range(len(ingredients_df.columns))
            
            for _, row in ingredients_df.iterrows():
                # Col 0 and 1 are item/quantity pair
                if pd.notna(row[0]):
                    recipe['ingredients'].append({
                        'item': str(row[0]).replace("\n", " ").replace("\\", " "),
                        'quantity': str(row[1]).replace("\n", " ").replace("\\", " ")
                    })
                # Col 2 and 3 are item/quantity pair
                if len(row) > 2 and pd.notna(row[2]):
                    recipe['ingredients'].append({
                        'item': str(row[2]).replace("\n", " ").replace("\\", " "),
                        'quantity': str(row[3]).replace("\n", " ").replace("\\", " ")
                    })
        return recipe
    except Exception as e:
        print(f"Error extracting ingredients: {e}")
        return None

def get_info(df):
    """Parses 2-column info tables."""
    res = pd.DataFrame()
    col_idx = 0
    for i in range(0, len(df.columns), 2):
        if i + 1 < len(df.columns):
            res[col_idx] = df.iloc[:, i].fillna(df.iloc[:, i + 1]).astype(str).str.replace("\n", " ").tolist()
        col_idx += 1
    
    if not res.empty:
        res.columns = res.iloc[0,:]
        res = res[1:]
    return res

def handle_table(table, idx, output_dir):
    """Decides how to process a table found in the PDF."""
    df = (pd.DataFrame(table)
        .map(lambda x: np.nan if x in ["", None] else x)
        .dropna(how="all", axis=0)
        .dropna(how="all", axis=1))
    
    df = df.reset_index(drop=True)
    df.columns = range(len(df.columns))
    
    if df.shape == (1,1) and df.iloc[0, 0]:
        return

    # Template 1: Recipe Ingredient Table
    if not df.empty and df.iloc[0,0] == "Name of dish":
        res = extract_ingredients(df)
        if res:
            dish_name = str(res["dish_name"]).replace(" ", "_").replace("/", "-")
            with open(output_dir / f"foc_ingredients_{dish_name}.json", "w", encoding='utf-8') as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
        return

    # Template 2: Info Lists
    elif not df.empty and pd.isna(df.iloc[0, 0]) and len(df.columns) > 1 and pd.isna(df.iloc[1, 1]):
        res = get_info(df)
        res.to_csv(output_dir / f"foc_table_{idx}.csv")
        return

def handle_text(text, output_path):
    """Extracts numbered sections and glossary from raw text using Regex."""
    # Remove page headers
    text = re.sub(r'^\d+FUNDAMENTALS OF COOKING\s*$', '', text, flags=re.MULTILINE)
    
    glossary_match = re.search(r'\bGlossary\b', text, re.IGNORECASE)
    pattern = r'(\d+(?:\.\d+)+)([^\n]+)'
    matches = list(re.finditer(pattern, text))
    stop_keywords = r'\b(Exercise|Activity|Teacher\'s guide|Glossary)\b'
    
    sections = {}
    
    # Extract numbered sections
    for i, match in enumerate(matches):
        section_title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end]
        
        stop_match = re.search(stop_keywords, content, re.IGNORECASE)
        if stop_match:
            content = content[:stop_match.start()]
        
        content = content.strip().replace('\n', ' ')
        content = re.sub(r' {2,}', ' ', content)
        if content:
            sections[section_title] = content
    
    # Extract Glossary
    if glossary_match:
        glossary_text = text[glossary_match.end():].strip()
        glossary_pattern = r'([A-Z][A-Za-z\s]+?)\s*[--]\s*((?:(?![A-Z][a-z]+\s*[--]).)+)'
        glossary = {}
        for match in re.finditer(glossary_pattern, glossary_text, re.DOTALL):
            term = match.group(1).strip()
            definition = match.group(2).strip().replace('\n', ' ')
            glossary[term] = re.sub(r' {2,}', ' ', definition)
        if glossary:
            sections['Glossary'] = glossary
            
    with open(output_path / "foc_sections.json", "w", encoding='utf-8') as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

# --- Main Processing Function ---

def process_pdfs(raw_data_path, output_path):
    raw_data_path = Path(raw_data_path)
    output_path = Path(output_path)
    foc_dir = output_path / "foc"
    foc_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing PDFs from {raw_data_path}...")

    # 1. Process Cooking Methods PDF (Table Extraction)
    methods_pdf = raw_data_path / "cooking methods.pdf"
    if methods_pdf.exists():
        all_tables = []
        with pdfplumber.open(methods_pdf) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        # Clean table data
                        table = [[txt.replace("\n", " ").replace("\uf0b7", " ") if txt else txt for txt in tbl] for tbl in table]
                        df = pd.DataFrame(table)
                        all_tables.append(df)
        
        if all_tables:
            combined_df = pd.concat(all_tables, ignore_index=True, sort=False).fillna(np.nan)
            
            # Shift columns to fix OCR alignment issues
            if 4 in combined_df.columns:
                combined_df[2] = combined_df[2].fillna(combined_df[3])
                combined_df[3] = combined_df[3].fillna(combined_df[4])
                combined_df.drop(4, axis=1, inplace=True)
            
            combined_df.columns = ["COOKING METHODS", "DESCRIPTION", "MERITS", "DEMERITS"]
            combined_df = combined_df[1:] # Drop header row
            
            # Merge rows split across pages
            combined_df = combined_df.fillna('')
            rows_to_combine = list(combined_df[combined_df["COOKING METHODS"] == ''].index)
            for row in rows_to_combine:
                for col in combined_df.columns:
                    prev_val = str(combined_df.loc[row - 1, col])
                    curr_val = str(combined_df.loc[row, col])
                    combined_df.loc[row - 1, col] = (prev_val + curr_val).strip()
            combined_df = combined_df.drop(rows_to_combine, axis=0)
            
            # Add "TYPE OF METHOD" column
            type_idx = list(combined_df[combined_df["DESCRIPTION"] == ''].index)
            # We initialize with None or an empty string to ensure it is treated as an 'object' (text) column
            combined_df['TYPE OF METHOD'] = None 
            combined_df.loc[combined_df.index.isin(type_idx), 'TYPE OF METHOD'] = combined_df.loc[combined_df.index.isin(type_idx), 'COOKING METHODS']
            combined_df['TYPE OF METHOD'] = combined_df['TYPE OF METHOD'].ffill()
            combined_df = combined_df.drop(type_idx, axis=0)
            
            combined_df.to_csv(output_path / "cooking_methods.csv", index=False)
            print("✓ Created cooking_methods.csv")
    else:
        print("x cooking methods.pdf not found.")

    # 2. Process Fundamentals PDF (Recipe & Text Extraction)
    foc_pdf = raw_data_path / "FundamentalsofCooking10.pdf"
    if foc_pdf.exists():
        all_text = ""
        with pdfplumber.open(foc_pdf) as pdf:
            i = 0
            for page in pdf.pages:
                # Handle Tables
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        handle_table(table, i, foc_dir)
                        i += 1
                
                # Handle Text
                text = page.extract_text()
                if text and "FUNDAMENTALS OF COOKING" in " ".join(text.split()[:5]):
                    all_text += text

        handle_text(all_text, output_path)
        print("✓ Processed Fundamentals of Cooking (Sections & Recipes)")
    else:
        print("x FundamentalsofCooking10.pdf not found.")

if __name__ == "__main__":
    process_pdfs("data/raw", "data/processed")