import os
import pandas as pd
from PyPDF2 import PdfReader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize embeddings and the vector store
db_location = "./chroma_langchain_db"


def reset_create_db():
    """Check if vector store needs to be rebuilt"""
    # If the directory doesn't exist, we definitely need to build it
    if not os.path.exists(db_location):
        print("Vector store directory doesn't exist. Creating and populating.")
        return True

    # If directory exists, check if it contains actual Chroma files
    chroma_files = ["chroma.sqlite3"]
    if not all(os.path.exists(os.path.join(db_location, f)) for f in chroma_files):
        print(
            "Vector store directory exists but is missing essential files. Rebuilding."
        )
        return True

    # Try to access the database
    try:
        # Check if any documents exist
        result = vector_store.get()
        if not result or "ids" not in result or len(result["ids"]) == 0:
            print("Vector store exists but is empty. Will populate.")
            return True
        else:
            doc_count = len(result["ids"])
            print(f"Vector store contains {doc_count} documents. Skipping rebuild.")
            return False
    except Exception as e:
        print(f"Error accessing vector store: {str(e)}. Rebuilding.")
        import shutil

        try:
            shutil.rmtree(db_location)
        except:
            print(f"Couldn't remove {db_location}. Please delete it manually.")
        return True


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

if not os.path.exists(db_location):
    os.makedirs(db_location)

try:
    # Try to load existing vector store
    vector_store = Chroma(
        collection_name="waterworks_documents",
        persist_directory=db_location,
        embedding_function=embeddings,
    )

    # Check if we need to populate it
    add_documents = reset_create_db()
except Exception as e:
    print(f"Error initializing vector store: {str(e)}. Will rebuild.")
    import shutil

    try:
        if os.path.exists(db_location):
            shutil.rmtree(db_location)
            os.makedirs(db_location)
    except:
        print(f"Couldn't reset {db_location}. Please check permissions.")

    # Create a new vector store
    vector_store = Chroma(
        collection_name="waterworks_documents",
        persist_directory=db_location,
        embedding_function=embeddings,
    )
    add_documents = True

vector_store = Chroma(
    collection_name="waterworks_documents",
    persist_directory=db_location,
    embedding_function=embeddings,
)


# Function to extract text from PDF using PyPDF2
def pdf_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = []
        for page in reader.pages:
            extracted = page.extract_text() or ""
            if extracted.strip():
                text.append(extracted)

        full_text = " ".join(text)
        if not full_text.strip():
            print(f"Warning: No text extracted from {pdf_path}")
            return "No readable text content found in this document."
        return full_text
    except Exception as e:
        print(f"Error exctracting text from {pdf_path}: {str(e)}")
        return f"Error extracting text: {str(e)}"


def chunk_text(text, metadata, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.create_documents(texts=[text], metadatas=[metadata])
    return chunks


# Populate documents
if add_documents:
    documents, ids = [], []

    # Process PDFs
    for folder in ["internal_docs", "product_docs"]:
        for file in os.listdir(folder):
            if file.lower().endswith(".pdf"):
                path = os.path.join(folder, file)
                content = pdf_text(path)
                metadata = {"source": folder, "filename": file}

                doc_chunks = chunk_text(content, metadata)
                documents.extend(doc_chunks)

                for i, _ in enumerate(doc_chunks):
                    ids.append(f"{folder}_{file}_{i}")

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
        "SKU-10": "PureWell Faucet",
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
                "customer_ID": row["Customer_ID"],
            },
        )
        documents.append(review_doc)
        ids.append(f"review_{idx}")

    # Add documents to Chroma in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        print(f"Adding batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        vector_store.add_documents(
            documents=documents[i : i + batch_size], ids=ids[i : i + batch_size]
        )

print("Chroma DB setup complete.")


def debug_vector_store():
    # Get all documents to check the count
    all_docs = vector_store.get()
    total_count = len(all_docs["ids"]) if "ids" in all_docs else 0
    print(f"Total documents in store: {total_count}")

    # Check document source distribution
    if "metadatas" in all_docs and all_docs["metadatas"]:
        sources = [m.get("source", "unknown") for m in all_docs["metadatas"] if m]
        source_count = {}
        for source in sources:
            source_count[source] = source_count.get(source, 0) + 1
        print("Document sources:", source_count)

    # Check for each expected SKU
    for sku in [f"SKU-{i}" for i in range(1, 11)]:
        try:
            results = vector_store.get(where={"SKU": sku})
            sku_count = len(results["ids"]) if "ids" in results else 0
            print(f"{sku}: Found {sku_count} documents")
        except Exception as e:
            print(f"Error checking {sku}: {str(e)}")

    # Check internal docs
    try:
        internal_docs = vector_store.get(where={"source": "internal_docs"})
        internal_count = len(internal_docs["ids"]) if "ids" in internal_docs else 0
        print(f"Internal docs: Found {internal_count} documents")
        if (
            "metadatas" in internal_docs
            and internal_docs["metadatas"]
            and internal_count > 0
        ):
            print(
                "Sample internal doc filenames:",
                [
                    m.get("filename", "unknown")
                    for m in internal_docs["metadatas"][:5]
                    if m
                ],
            )
    except Exception as e:
        print(f"Error checking internal docs: {str(e)}")

    # Check product docs
    try:
        product_docs = vector_store.get(where={"source": "product_docs"})
        product_count = len(product_docs["ids"]) if "ids" in product_docs else 0
        print(f"Product docs: Found {product_count} documents")
        if (
            "metadatas" in product_docs
            and product_docs["metadatas"]
            and product_count > 0
        ):
            print(
                "Sample product doc filenames:",
                [
                    m.get("filename", "unknown")
                    for m in product_docs["metadatas"][:5]
                    if m
                ],
            )
    except Exception as e:
        print(f"Error checking product docs: {str(e)}")


debug_vector_store()
