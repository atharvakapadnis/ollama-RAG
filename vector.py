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

vector_store = Chroma(
    collection_name = "waterworks_reviews",
    persist_directory = db_location,
    embedding_function = edmbeddings
)

if add_documents:
    vector_store.add_documents(documents = documents, ids = ids)

retriever = vector_store.as_restriever(
    search_kwargs={"k":50}
)