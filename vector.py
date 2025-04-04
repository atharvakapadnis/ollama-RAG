import os
import pandas as pd
from PyPDF2 import PdfReader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Initialize embeddings and the vector store
edmbeddings = OllamaEmbeddings(model = "mxbai-embed-large")
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

vector_store = Chroma(
    collection_name = "waterworks_documents",
    persist_directory = db_location,
    embedding_function = edmbeddings
)

# Function to extract text from PDF using PyPDF2
def pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    return " ".join(page.extract_text() or "" for page in reader.pages)

# Populate documents
if add_documents:
    documents, ids = [], []

    # Process PDFs
    for folder in ['internal_docs', 'product_docs']:
        for file in os.listdir(folder):
            if file.lower().endswith(".pdf"):
                path = os.path.join(folder, file)
                content = pdf_text(path)
                document = Document(
                    page_content = content,
                    metadata={"source": folder, "filename": file}
                )
                documents.append(document)
                ids.append(f"{folder}_{file}")

    # Explicitly define relationship among SKUs and docs
    sku_mapping = {
    "SKU-1": "AquaFlow Pro",
    "SKU-2": "PureStream Deluxe",
    "SKU-3": "HydroMax Ultra",
    "SKU-4": "EcoPure Bottle",
    "SKU-5": "StreamLine Valve",
    "SKU-6": "ClearWave Filter",
    "SKU-7": "AquaGuard Sensor",
    "SKU-8": "RainSaver Barrel",
    "SKU-9": "ThermoFlow Heater",
    "SKU-10": "PureWell Faucet"
    }

    # Updated CSV ingestion
    df = pd.read_csv("customer_reviews/customer_reviews.csv")
    for idx, row in df.iterrows():
        sku = row["SKU"]
        product_name = sku_mapping.get(sku, "Unknown Product")
        review_doc = Document(
            page_content=f"{product_name} ({sku}) - {row['Review_Text']}",
            metadata={
                "source": "reviews",
                "SKU": sku,
                "product_name": product_name,
                "rating": row["Rating"],
                "date": row["Date"],
                "customer_ID": row["Customer_ID"]
            }
        )
        documents.append(review_doc)
        ids.append(f"review_{idx}")

    # Add documents to Chroma in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        print(f"Adding batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        vector_store.add_documents(documents=documents[i:i+batch_size], ids=ids[i:i+batch_size])

print("Chroma DB setup complete.")