from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import vector_store

# Initialize the Ollama Model
model = OllamaLLM(model="llama3.2")

# Setup Retriever function
product_retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 10,
        "filter": {"source": {"$eq": "product_docs"}},
    }
)

review_retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 10,
        "filter": {"source": {"$eq": "reviews"}},
    }
)

internal_retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 10,
        "filter": {"source": {"$eq": "internal_docs"}},
    }
)

# Define template for context handling
template = """
You are an expert assistant specialized in a Water Works Supply Chain business. Your role is to provide clear, detailed, and accurate answers based on the information provided.

### CRITICAL INSTRUCTION:
The following SKU mapping is DEFINITIVE and ALWAYS CORRECT, regardless of what you find in the documents:
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

You MUST use this mapping when asked about SKUs. If asked to list all SKUs, ALWAYS list ALL TEN of them with their names, even if they aren't mentioned in the context.

### Query Type Detection:
- For questions about company policies, HR, benefits, or internal operations, prioritize information from Internal Company Documents.
- For questions about specific products or SKUs, prioritize Product Documentation, followed by Customer Reviews.
- For questions about customer satisfaction or product feedback, prioritize Customer Reviews.

### Instructions for Answering:
- Clearly differentiate whether your answer is based on Internal Company Documents, Product Documentation, or Customer Reviews.
- Always mention the specific document name when referencing information from internal documents or product documentation.
- If multiple sources provide conflicting information, mention each viewpoint clearly and separately.
- Include specific details such as product names (SKUs), ratings, policies, or processes when relevant.
- If the provided context doesn't contain enough information, honestly state that and avoid speculation.
- IMPORTANT: Always scan the entire context for relevant information before responding.

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
    # Group documents by source
    sources = {}
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if source not in sources:
            sources[source] = []
        sources[source].append(doc)

    # Format context with clear source sections
    formatted_context = ""

    # Put internal docs first
    if "internal_docs" in sources:
        formatted_context += "\n=== INTERNAL COMPANY DOCUMENTS ===\n"
        for doc in sources["internal_docs"]:
            filename = doc.metadata.get("filename", "unknown")
            formatted_context += f"\n[{filename}]:\n{doc.page_content}\n"

    # Then product docs
    if "product_docs" in sources:
        formatted_context += "\n=== PRODUCT DOCUMENTATION ===\n"
        for doc in sources["product_docs"]:
            filename = doc.metadata.get("filename", "unknown")
            formatted_context += f"\n[{filename}]:\n{doc.page_content}\n"

    # Then reviews, grouped by SKU
    if "reviews" in sources:
        formatted_context += "\n=== CUSTOMER REVIEWS ===\n"
        # Group reviews by SKU
        sku_reviews = {}
        for doc in sources["reviews"]:
            sku = doc.metadata.get("SKU", "unknown")
            if sku not in sku_reviews:
                sku_reviews[sku] = []
            sku_reviews[sku].append(doc)

        # Format each SKU's reviews
        for sku, reviews in sku_reviews.items():
            product_name = reviews[0].metadata.get("product_name", "Unknown Product")
            formatted_context += f"\n[{product_name} ({sku})]:\n"
            for review in reviews[:5]:  # Limit to 5 reviews per SKU to save space
                rating = review.metadata.get("rating", "")
                formatted_context += f"- Rating {rating}: {review.page_content}\n"

    return formatted_context


def get_relevant_documents(question):
    """Retrieve documents with a more balanced and adaptive approach"""
    question_lower = question.lower()
    documents = []

    # Determine the focus of the question
    product_focus = 0
    internal_focus = 0
    review_focus = 0

    # Score the question based on keywords
    product_terms = [
        "sku",
        "product",
        "item",
        "specifications",
        "specs",
        "features",
        "material",
        "dimensions",
    ]
    internal_terms = [
        "policy",
        "leave",
        "vacation",
        "company",
        "internal",
        "procedure",
        "employee",
        "staff",
        "hr",
    ]
    review_terms = [
        "review",
        "customer",
        "satisfaction",
        "rating",
        "complaint",
        "feedback",
        "opinion",
    ]

    # Calculate weighted scores
    for term in product_terms:
        if term in question_lower:
            product_focus += 1

    for term in internal_terms:
        if term in question_lower:
            internal_focus += 1

    for term in review_terms:
        if term in question_lower:
            review_focus += 1

    # Normalize scores to determine document allocation
    total_score = max(1, product_focus + internal_focus + review_focus)
    total_docs = 20  # Maximum number of docs to retrieve

    # Calculate document proportions (minimum 2 if score > 0)
    product_docs_count = (
        max(2, int(total_docs * product_focus / total_score))
        if product_focus > 0
        else 3
    )
    internal_docs_count = (
        max(2, int(total_docs * internal_focus / total_score))
        if internal_focus > 0
        else 3
    )
    review_docs_count = (
        max(2, int(total_docs * review_focus / total_score)) if review_focus > 0 else 3
    )

    # For generic questions with no specific focus, use a balanced approach
    if product_focus == 0 and internal_focus == 0 and review_focus == 0:
        product_docs_count = 5
        internal_docs_count = 5
        review_docs_count = 5

    # Get internal docs
    internal_docs = internal_retriever.invoke(question)
    documents.extend(internal_docs[:internal_docs_count])

    # Get product docs
    product_docs = product_retriever.invoke(question)
    documents.extend(product_docs[:product_docs_count])

    # Get reviews
    reviews = review_retriever.invoke(question)
    documents.extend(reviews[:review_docs_count])

    # If asking explicitly about specific SKUs, add targeted reviews
    for i in range(1, 11):
        sku = f"SKU-{i}"
        if sku.lower() in question_lower:
            # Get reviews specific to this SKU
            sku_reviews = review_retriever.invoke(sku)
            # Add up to 3 SKU-specific reviews
            documents.extend(sku_reviews[:3])

    return documents


# Main interaction loop
if __name__ == "__main__":
    print("Agent Ready. Ask your questions!")

    while True:
        question = input("\n Enter your question (type q to quit):")
        if question.lower() == "q":
            print("Terminating Process")
            break

        docs = get_relevant_documents(question)
        context = format_context(docs)
        result = chain.invoke({"context": context, "question": question})

        print("\n Answer:")
        print(result)
