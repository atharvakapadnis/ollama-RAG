import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

df = pd.read_csv("synthetic_customer_reviews_expanded.csv")

edmbeddings = OllamaEmbeddings(model = "mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content  = row["SKU"]+" "+row["Review_Text"],
            metadata = {"rating": row["Rating"], "date": row["Date"], "customer_ID": row["Customer_ID"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

def doc_batches(vector_store, documents, ids, batch_size=5000):
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        print(f"Adding batch {i//batch_size + 1}/{(len(documents)-1)//batch_size+1} with {len(batch_docs)} documents")
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
    return vector_store

vector_store = Chroma(
    collection_name = "waterworks_reviews",
    persist_directory = db_location,
    embedding_function = edmbeddings
)

if add_documents:
    doc_batches(vector_store, documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k":1000}
)