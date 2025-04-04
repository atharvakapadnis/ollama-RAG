# Water Works Supply Chain RAG Application

A specialized question-answering system built for the water works supply chain industry that leverages the Retrieval-Augmented Generation (RAG) approach to provide accurate responses about business operations, customer reviews, and product information.

## Overview

This application allows you to query a database of customer reviews related to water works supply chain products and receive contextually relevant answers from an LLM. The system uses:

- **Ollama** for local LLM inference (llama3.2 model)
- **LangChain** for the orchestration framework
- **Chroma DB** as the vector database for semantic search
- **RAG (Retrieval-Augmented Generation)** methodology to enhance responses with relevant customer review data

## Features

- Interactive question-answering interface
- Semantic search through customer review data
- Contextually-aware responses powered by the llama3.2 model
- Local inference with no data sent to external APIs
- Specialized for water works supply chain business domain

## Prerequisites

- Python 3.8+
- Ollama installed locally with llama3.2 model available
- Sufficient storage space for the vector database

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/atharvakapadnis/water-works-rag.git
   cd water-works-rag
   ```

2. Install the required packages:
   ```bash
   pip install langchain langchain-ollama langchain-chroma pandas
   ```

3. Make sure you have Ollama installed and the required models:
   ```bash
   # Install llama3.2 model
   ollama pull llama3.2
   
   # Install embedding model
   ollama pull mxbai-embed-large
   ```

4. Ensure you have the `synthetic_customer_reviews_expanded.csv` file in the root directory of the project.

## Usage

Run the main application:
```bash
python main.py
```

This will start an interactive console where you can ask questions about the water works supply chain business. Type 'q' to exit the application.

Example questions:
- "What are common complaints about valve products?"
- "What products have the highest customer satisfaction?"
- "Are there any supply chain issues mentioned in recent reviews?"

## How It Works

1. **Vector Database Setup**: The `vector.py` file processes customer review data from the CSV file and creates embeddings using the mxbai-embed-large model. These embeddings are stored in a Chroma vector database.

2. **Retrieval**: When you ask a question, the system converts it to an embedding and finds the most semantically similar customer reviews in the database.

3. **Generation**: The relevant reviews are sent along with your question to the llama3.2 model, which generates a contextually informed response.

## Project Structure

- `main.py`: Main application entry point with the interactive CLI
- `vector.py`: Setup for the vector database and retrieval functionality
- `synthetic_customer_reviews_expanded.csv`: Dataset containing customer reviews (not included in repo)
- `chroma_langchain_db/`: Directory where the vector database is stored (generated on first run)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Atharva Vishal Kapadnis, 2025

## Acknowledgments

- This project uses [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- Local inference is powered by [Ollama](https://github.com/ollama/ollama)
- Vector database functionality provided by [Chroma](https://github.com/chroma-core/chroma)
