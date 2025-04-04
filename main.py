from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import vector_store

# Initialize the Ollama Model
model = OllamaLLM(model="llama3.2")

# Setup Retriever function
retriever = vector_store.as_retriever(search_kwargs={
    "k": 50,
    "filter": {"source": {"$in": ["internal_docs", "product_docs"]}}
})

# Define template for context handling
template = """
You are an expert assistant specialized in a Water Works Supply Chain business. Your role is to provide clear, detailed, and accurate answers based on the information provided.

### SKU Identifier Mapping:
Each SKU identifier from customer reviews exactly matches the following products in the product documentation:

- SKU-1: AquaFlow Pro  
- SKU-2: PureStream Deluxe  
- SKU-3: HydroMax Ultra  
- SKU-4: EcoPure Bottle  
- SKU-5: StreamLine Valve  
- SKU-6: ClearWave Filter  
- SKU-7: AquaGuard Sensor  
- SKU-8: RainSaver Barrel  
- SKU-9: ThermoFlow Heater  
- SKU-10: PureWell Faucet  

Use this mapping when referencing SKU identifiers from reviews and documentation.

### Instructions for Answering:
- Clearly differentiate whether your answer is based on Internal Company Documents, Product Documentation, or Customer Reviews.
- If multiple sources provide conflicting information, mention each viewpoint clearly and separately.
- Include specific details such as product names (SKUs), ratings, policies, or processes when relevant.
- If the provided context doesn't contain enough information, honestly state that and avoid speculation.

### Provided Context:
{context}

### User Question:
{question}

### Expert Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Format context clearly for the model
def format_context(docs):
    formatted_context = "" 
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        filename = doc.metadata.get("filename", "")
        rating = doc.metadata.get("rating", "")
        sku = doc.metadata.get("SKU", "")

        if source in ["internal_docs", "product_docs"]:
            formatted_context += f"\n [{source}/{filename}]: {doc.page_content[:500]}...\n"
        elif source == "reviews":
            formatted_context += f"\n [Review | SKU: {sku} | Rating: {rating}]: {doc.page_content}\n"
    return formatted_context

# Main interaction loop
if __name__ == "__main__":
    print("Agent Ready. Ask your questions!")

    while True:
        question = input("\n Enter your question (type q to quit):")
        if question.lower() == "q":
            print("Terminatin Process")
            break

        docs = retriever.invoke(question)
        context = format_context(docs)
        result = chain.invoke({"context": context, "question": question})

        print("\n Answer:")
        print(result)