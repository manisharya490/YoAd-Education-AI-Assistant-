# YoAd RAG System

A Retrieval-Augmented Generation (RAG) system that connects a custom knowledge base with a Large Language Model to create an intelligent Q&A interface.

## Project Structure

```
jarvis/
├── data/               # Store your custom documents here
├── ingestion.py       # Data ingestion pipeline
├── jarvis_app.py      # Main RAG application
├── requirements.txt   # Project dependencies
└── .env              # Environment variables
```

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Ollama and download the Llama2 model:
```bash
# Follow Ollama installation instructions from: https://ollama.ai/
ollama pull llama2
```

3. Configure environment variables:
- Copy `.env.example` to `.env`
- Add your Pinecone API key and environment details

4. Prepare your data:
- Place your custom documents in the `data/` directory
- Supported formats: PDF, TXT, MD, etc.

5. Run the ingestion pipeline:
```bash
python ingestion.py
```

6. Start the YoAd RAG system:
```bash
python jarvis_app.py
```

## System Components

### Data Ingestion Pipeline (`ingestion.py`)
- Document loading from local directory
- Text splitting with RecursiveCharacterTextSplitter
- Embedding generation using BAAI/bge-small-en-v1.5
- Vector storage in Pinecone

### RAG Application (`jarvis_app.py`)
- Local LLM integration using Ollama (Llama2)
- Custom RAG prompt template
- Pinecone retriever with top-3 similarity search
- Interactive CLI interface

## Testing

Test the system with different types of queries:
1. Invention-specific questions (should be answered from your documents)
2. General knowledge questions (should indicate information not in knowledge base)
3. Questions requiring information synthesis from multiple chunks

## Notes

- The system uses the BAAI/bge-small-en-v1.5 embedding model (384 dimensions)
- Chunks are created with 1000-character size and 200-character overlap
- The retriever fetches the top 3 most relevant document chunks
- Local LLM deployment using Ollama for privacy and cost-effectiveness