from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about a Water Works Based Supply Chain Business

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

result = chain.invoke({"reviews": [], "question": "What is the best product this company has?"})
print(result)