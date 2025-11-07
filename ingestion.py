import os
import hashlib
import json
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
import torch


class LocalHFEmbeddings:
    """Simple HF-transformers based embedding with mean pooling to avoid sentence-transformers dependency."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
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
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_CLOUD")
if not PINECONE_API_KEY or not PINECONE_ENV:
    print("Warning: Pinecone API key or environment not set in .env. Please populate PINECONE_API_KEY and PINECONE_ENVIRONMENT before running ingestion.")

# Initialize Pinecone client only when credentials exist (safe to import file without credentials)
pc = None
if PINECONE_API_KEY and PINECONE_ENV:
    pc = Pinecone(api_key=PINECONE_API_KEY)


class DataIngestionPipeline:
    def __init__(self, data_dir: str = "./data", index_name: str = "yoad-index"):
        self.data_dir = data_dir
        self.index_name = index_name
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        # Use a robust, commonly available embedding model by default (local HF implementation)
        # Prefer the local wrapper which avoids sentence-transformers dependency.
        try:
            self.embeddings = LocalHFEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            # fallback to langchain wrapper if available in environment
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # manifest file to track file -> chunk ids mapping for incremental updates
        self.manifest_path = os.path.join(os.getcwd(), "ingestion_manifest.json")

    def _get_loader_for_file(self, path: str):
        lower = path.lower()
        if lower.endswith(".pdf"):
            return PyPDFLoader(path)
        if lower.endswith(".pptx") or lower.endswith(".ppt"):
            return UnstructuredPowerPointLoader(path)
        if lower.endswith(".docx") or lower.endswith(".doc"):
            return UnstructuredWordDocumentLoader(path)
        if lower.endswith(".txt"):
            return TextLoader(path, encoding="utf8")
        # fallback to unstructured generic loader
        return UnstructuredFileLoader(path)

    def load_documents(self) -> List:
        """Load documents from the data directory using appropriate loaders per file type."""
        docs = []
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        for root, _, files in os.walk(self.data_dir):
            for fname in files:
                path = os.path.join(root, fname)
                try:
                    loader = self._get_loader_for_file(path)
                    loaded = loader.load()
                    # attach source metadata
                    for d in loaded:
                        if not getattr(d, "metadata", None):
                            d.metadata = {}
                        d.metadata["source"] = path
                    docs.extend(loaded)
                except Exception as e:
                    print(f"Failed to load {path}: {e}")

        print(f"Loaded {len(docs)} documents from {self.data_dir}")
        return docs

    def split_documents(self, documents: List):
        print("Splitting documents into chunks...")
        return self.text_splitter.split_documents(documents)

    def _ensure_index(self, dimension: int):
        if not (PINECONE_API_KEY and PINECONE_ENV) or not pc:
            raise RuntimeError("Pinecone credentials are missing; cannot create or connect to index.")

        # Check existing indexes and create if necessary
        existing_names = pc.list_indexes().names()
        if self.index_name not in existing_names:
            raise RuntimeError(
                f"Pinecone index '{self.index_name}' not found in your account.\n"
                f"Please create an index named '{self.index_name}' with dimension={dimension} in the Pinecone console "
                f"or provide a valid environment/region.\nExample: create a Serverless index with the correct cloud/region in the Pinecone dashboard."
            )

        return pc.Index(self.index_name)

    def ingest_data(self, batch_size: int = 64):
        """Full ingestion pipeline: load, split, embed (batch), deduplicate by hash, upsert into Pinecone.

        This implementation supports incremental updates by keeping a JSON manifest mapping
        source file -> list of chunk ids (SHA256 of chunk text). On each run it computes
        which chunk ids are new (to embed & upsert) and which were removed (to delete from
        Pinecone). After the run the manifest is updated.
        """
        # Load previous manifest (if any)
        prev_manifest = self._load_manifest()

        docs = self.load_documents()
        if not docs:
            print("No documents found to ingest.")
            return 0

        chunks = self.split_documents(docs)
        print(f"Created {len(chunks)} chunks from documents")

        # Build current manifest and mapping of id -> (text, meta)
        current_manifest = {}
        id_to_text = {}
        id_to_meta = {}

        for chunk in chunks:
            text = chunk.page_content
            _id = hashlib.sha256(text.encode("utf-8")).hexdigest()
            src = chunk.metadata.get("source", "unknown")
            current_manifest.setdefault(src, []).append(_id)
            id_to_text[_id] = text
            id_to_meta[_id] = {"source": src, "text": text}

        # Determine deleted ids (files removed entirely or chunks removed)
        deleted_ids = []
        for prev_src, prev_ids in prev_manifest.items():
            if prev_src not in current_manifest:
                # entire file removed
                deleted_ids.extend(prev_ids)
            else:
                # file exists, determine removed chunk ids
                removed = set(prev_ids) - set(current_manifest.get(prev_src, []))
                deleted_ids.extend(list(removed))

        # Determine ids to upsert (new chunks)
        to_upsert_ids = []
        for src, ids in current_manifest.items():
            prev_ids = set(prev_manifest.get(src, []))
            for _id in ids:
                if _id not in prev_ids:
                    to_upsert_ids.append(_id)

        total_upserts = len(to_upsert_ids)
        print(f"Detected {total_upserts} new chunks to upsert and {len(deleted_ids)} chunks to delete.")

        index = None
        # If we need to embed new text, create/get index with correct dimension
        if total_upserts > 0:
            sample = self.embeddings.embed_query("hello")
            dimension = len(sample)
            index = self._ensure_index(dimension)

        # If only deletions and no new embeddings, connect to existing index to delete ids
        if index is None and deleted_ids:
            if not (PINECONE_API_KEY and PINECONE_ENV) or not pc:
                raise RuntimeError("Pinecone credentials are missing; cannot delete vectors from index.")
            existing_names = pc.list_indexes().names()
            if self.index_name not in existing_names:
                print(f"Index '{self.index_name}' not found; nothing to delete.")
                index = None
            else:
                index = pc.Index(self.index_name)

        # Delete old vectors if any
        if index is not None and deleted_ids:
            print(f"Deleting {len(deleted_ids)} stale vectors from index...")
            # Pinecone delete accepts ids list
            index.delete(ids=deleted_ids)

        # Upsert new vectors in batches
        if total_upserts > 0 and index is not None:
            all_new_ids = to_upsert_ids
            print(f"Embedding {len(all_new_ids)} new chunks in batches of {batch_size}...")
            for i in tqdm(range(0, len(all_new_ids), batch_size)):
                batch_ids = all_new_ids[i : i + batch_size]
                batch_texts = [id_to_text[_id] for _id in batch_ids]
                batch_metas = [id_to_meta[_id] for _id in batch_ids]

                embeddings = self.embeddings.embed_documents(batch_texts)

                vectors = []
                for _id, emb, meta, txt in zip(batch_ids, embeddings, batch_metas, batch_texts):
                    meta_with_text = dict(meta)
                    meta_with_text["text"] = txt
                    vectors.append((_id, emb, meta_with_text))

                index.upsert(vectors=vectors)

        # Save the new manifest
        self._save_manifest(current_manifest)

        print(f"Ingestion complete. Upserted {total_upserts} new chunks; deleted {len(deleted_ids)} old chunks.")
        return total_upserts

    def _load_manifest(self):
        if not os.path.exists(self.manifest_path):
            return {}
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_manifest(self, manifest: dict):
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            print(f"Failed to save manifest: {e}")


if __name__ == "__main__":
    # Debug: Show Pinecone environment variables
    print(f"Using Pinecone environment: {os.getenv('PINECONE_ENVIRONMENT')}")
    print(f"Index host: {os.getenv('PINECONE_INDEX_HOST')}")
    print(f"API Key configured: {'Yes' if os.getenv('PINECONE_API_KEY') else 'No'}")
    
    pipeline = DataIngestionPipeline()
    try:
        print("\nStarting ingestion...")
        count = pipeline.ingest_data()
        print(f"Ingestion complete, upserted {count} vectors.")
    except Exception as e:
        print(f"Ingestion failed: {e}")