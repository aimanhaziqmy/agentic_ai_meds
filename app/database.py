import os
import pandas as pd
import chromadb
from chromadb.config import Settings

# Ensure the DB directory exists
DB_PATH = "./chroma_db"
os.makedirs(DB_PATH, exist_ok=True)

client = chromadb.PersistentClient(path=DB_PATH)

def initialize_databases():
    """Reads CSVs from the /data folder and ingests them into ChromaDB."""

    # --- 3. Drugs Dataset ---
    collection_drugs = client.get_or_create_collection(name="drugs_dataset")
    if collection_drugs.count() == 0 and os.path.exists("./data/drugs_dataset.csv"):
        print("Ingesting Drugs Dataset...")
        df_drugs = pd.read_csv("./data/drugs_dataset.csv").dropna(subset=['Medicine Name']).head(500)
        df_drugs['combined_text'] = (
            "Medicine Name: " + df_drugs['Medicine Name'].astype(str) + ". " +
            "Composition: " + df_drugs.get('Composition', '').astype(str) + ". " +
            "Uses: " + df_drugs.get('Uses', '').astype(str) + ". " +
            "Side Effects: " + df_drugs.get('Side_effects', '').astype(str) + ". " +
            "Manufacturer: " + df_drugs.get('Manufacturer', '').astype(str) + ". " +
            "Reviews (Excellent/Avg/Poor): " + df_drugs.get('Excellent Review', '0').astype(str) + "% / " + 
            df_drugs.get('Average Review', '0').astype(str) + "% / " + df_drugs.get('Poor Review', '0').astype(str) + "%."
        )
        collection_drugs.add(
            documents=df_drugs['combined_text'].tolist(),
            metadatas=df_drugs.drop(columns=['combined_text']).to_dict(orient='records'),
            ids=[f"drug_{i}" for i in df_drugs.index.tolist()]
        )

    # --- 2. Medical QnA Dataset ---
    collection_qna = client.get_or_create_collection(name="medical_qna")
    if collection_qna.count() == 0 and os.path.exists("./data/medical_qna_dataset.csv"):
        print("Ingesting QnA Dataset...")
        df_qa = pd.read_csv("./data/medical_qna_dataset.csv").dropna(subset=['Question', 'Answer']).head(500)
        df_qa['combined_text'] = (
            "Question: " + df_qa['Question'].astype(str) + '. ' +
            "Answer: " + df_qa['Answer'].astype(str) + '. ' +
            "Type: " + df_qa['qtype'].astype(str) + '.'
        )
        collection_qna.add(
            documents=df_qa['combined_text'].tolist(),
            metadatas=df_qa.drop(columns=['combined_text']).to_dict(orient='records'),
            ids=df_qa.index.astype(str).tolist()
        )

    # --- 3. Medical Device Manuals ---
    collection_device = client.get_or_create_collection(name="medical_device_manual")
    if collection_device.count() == 0 and os.path.exists("./data/medical_device_manuals_dataset.csv"):
        print("Ingesting Device Manuals Dataset...")
        df_md = pd.read_csv("./data/medical_device_manuals_dataset.csv").dropna(subset=['Device_Name']).head(500)
        df_md['combined_text'] = (
            "Device Name: " + df_md['Device_Name'].astype(str) + ". " +
            "Model: " + df_md.get('Model_Number', '').astype(str) + ". " +
            "Manufacturer: " + df_md.get('Manufacturer', '').astype(str) + ". " +
            "Indications: " + df_md.get('Indications_for_Use', '').astype(str) + ". " +
            "Contraindications: " + df_md.get('Contraindications', 'None').astype(str)
        )
        collection_device.add(
            documents=df_md['combined_text'].tolist(),
            metadatas=df_md.drop(columns=['combined_text']).to_dict(orient='records'),
            ids=[f"dev_{i}" for i in df_md.index.tolist()]
        )



    return collection_qna, collection_device, collection_drugs

# Make it can be imported globally
collection_qna, collection_device, collection_drugs = initialize_databases()
