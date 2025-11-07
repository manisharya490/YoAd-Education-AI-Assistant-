import os
import textwrap
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, pipeline
import torch
try:
    import ollama
    HAS_OLLAMA = True
except Exception:
    HAS_OLLAMA = False


class LocalHFEmbeddings:
    """Simple embeddings implementation using a HF transformer encoder with mean pooling.

    Methods:
    - embed_documents(list[str]) -> list[list[float]]
    - embed_query(str) -> list[float]
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Use AutoModel (encoder) for embeddings
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_documents(self, texts: list) -> list:
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for t in texts:
                encoded = self.tokenizer(t, truncation=True, padding=True, return_tensors="pt")
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                model_output = self.model(**encoded)
                emb = self._mean_pooling(model_output, encoded["attention_mask"])
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                embeddings.append(emb[0].cpu().numpy().tolist())
        return embeddings

    def embed_query(self, text: str) -> list:
        return self.embed_documents([text])[0]
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize Pinecone client
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set in .env")

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = os.getenv("PINECONE_INDEX", "yoad-index")


class YoAdRAG:
    def __init__(self, index_name: str = INDEX_NAME, top_k: int = 3):
        print("Initializing YoAd RAG system...")
        print("Loading embedding model...")
        # Embedding model (transformers-based local implementation)
        self.embeddings = LocalHFEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Pinecone index handle
        self.index = pc.Index(index_name)
        self.top_k = top_k

        print("Loading language model (this may take a minute)...")
        # Prefer Ollama if available and model pulled locally
        self.use_ollama = HAS_OLLAMA
        if self.use_ollama:
            self.ollama_model = os.getenv("OLLAMA_MODEL", "llama2")
            print(f"Using Ollama model: {self.ollama_model}")
            self.llm = None
        else:
            # Initialize local LLM using a lightweight model as fallback
            model_id = "facebook/opt-125m"
            pipe = pipeline(
                "text-generation",
                model=model_id,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)

        # Prompt template (simple format string)
        self.prompt_template = textwrap.dedent("""
        You are YoAd, a helpful, precise, and polite AI assistant. Answer the user's question based solely on the provided context from the internal knowledge base.
        If the context does not contain the relevant information, respond exactly with: "The required information is not available in the internal knowledge base."

        Context:
        {context}

        Question: {question}

        Answer:
        """)

    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve top-k chunks from Pinecone for the query."""
        # Embed the query
        q_emb = self.embeddings.embed_query(query)

        # Query Pinecone
        resp = self.index.query(vector=q_emb, top_k=self.top_k, include_metadata=True)

        matches = []
        for m in getattr(resp, "matches", []) or resp.get("matches", []):
            metadata = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {})
            text = metadata.get("text") or metadata.get("source") or ""
            score = m.get("score") if isinstance(m, dict) else getattr(m, "score", None)
            matches.append({"id": m.get("id") if isinstance(m, dict) else getattr(m, "id", None), "text": text, "score": score})

        return matches

    def answer(self, question: str) -> str:
        # Retrieve context
        matches = self.retrieve(question)

        if not matches:
            context = ""
        else:
            # Build context string from retrieved chunks
            pieces = []
            for i, m in enumerate(matches, 1):
                pieces.append(f"[{i}] {m['text']}")
            context = "\n\n".join(pieces)

        prompt = self.prompt_template.format(context=context, question=question)

        # If context is empty, short-circuit to ensure LLM returns the required message
        if not context.strip():
            return "The required information is not available in the internal knowledge base."

        # If Ollama is available, call it; otherwise use HF pipeline
        if self.use_ollama:
            try:
                gen = ollama.generate(model=self.ollama_model, prompt=prompt)
                # Ollama's response text can be in .response or str(gen)
                text = getattr(gen, "response", None)
                if text is None:
                    return str(gen)
                return text
            except Exception as e:
                print(f"Ollama error: {e}")
                return "I encountered an error while generating the answer with Ollama."
        else:
            try:
                response = self.llm.predict(prompt)
                if not response.strip():
                    return "I encountered an error while generating the answer."
                return response
            except Exception as e:
                print(f"Language model error: {e}")
                return "I encountered an error while generating the answer."


def main():
    yoad = YoAdRAG()
    print("YoAd is ready. Type a question or 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        ans = yoad.answer(q)
        print("\nYoAd:", ans)


if __name__ == "__main__":
    main()