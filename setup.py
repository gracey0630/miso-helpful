# setup.py
import os
os.system('chcp 65001 >nul 2>&1')  # Set console to UTF-8 to handle emojis
from src.data_processing.extract_flavor import process_flavor_network
from src.data_processing.extract_pdf import process_pdfs
from src.rag.database import CookingDB

def run_setup():
    print("🚀 Starting misohelpful project setup...")
    
    # 1. Create directory structure
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # 2. Run Data Extraction
    print("\n--- Extracting Data ---")
    process_flavor_network("data/raw", "data/processed")
    process_pdfs("data/raw", "data/processed")
    
    # 3. Initialize and Ingest Vector Database
    print("\n--- Building Vector Database ---")
    db = CookingDB()
    db.ingest_data("data/processed")
    
    print("\n✅ Setup complete! Run 'streamlit run app.py' to start.")

if __name__ == "__main__":
    run_setup()