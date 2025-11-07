import os
import textwrap
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, pipeline
import torch
import speech_recognition as sr
import pyttsx3
try:
    import ollama
    HAS_OLLAMA = True
except Exception:
    HAS_OLLAMA = False

# Initialize speech recognition and text-to-speech engines
recognizer = sr.Recognizer()
engine = pyttsx3.init()


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
        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        # Set properties (optional)
        self.engine.setProperty('rate', 150)    # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        # Initialize recognizer for voice input
        self.recognizer = sr.Recognizer()

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

        # Prompt template instructing the LLM to synthesize an answer from retrieved context
        # It asks for a concise, directly useful answer and a short sources section.
        self.prompt_template = textwrap.dedent("""
        You are YoAd, a helpful, precise, and polite AI assistant. Using ONLY the information in the provided CONTEXT, synthesize a concise, factual answer to the user's question.

        - If the question is comparative (e.g. differences between libraries), provide a short, structured comparison.
        - Keep the answer concise (approx. 2-6 sentences) and avoid filler.
        - After the answer include a short 'Sources:' section listing the chunk numbers you used in the form [1], [2], ...
        - If the context does NOT contain the required information, output EXACTLY the single line:
          The required information is not available in the internal knowledge base.

        CONTEXT:
        {context}

        QUESTION: {question}

        OUTPUT FORMAT (must follow):
        Answer:
        <concise synthesized answer>

        Sources:
        [1] [2] ...
        """)

    def speak(self, text: str) -> None:
        """Speak the given text using text-to-speech."""
        self.engine.say(text)
        self.engine.runAndWait()

    def _extract_answer_text(self, llm_text: str) -> str:
        """Extract the 'Answer' section from the model output.

        If the output follows the requested format (Answer: ... \n\nSources: ...)
        this returns the answer text only. Otherwise returns the whole text.
        """
        if not llm_text:
            return ""
        # Normalize and search for markers
        txt = llm_text
        # Try to find 'Answer:' and 'Sources:' markers
        lower = txt.lower()
        ans_idx = lower.find('answer:')
        src_idx = lower.find('\nsources:')
        if ans_idx != -1:
            start = ans_idx + len('answer:')
            if src_idx != -1 and src_idx > ans_idx:
                answer = txt[start:src_idx]
            else:
                answer = txt[start:]
            return answer.strip()
        # fallback: try to split by two newlines then take first paragraph
        parts = txt.split('\n\n')
        return parts[0].strip() if parts else txt.strip()

    def listen(self) -> str:
        """Listen for voice input and return the recognized text."""
        with sr.Microphone() as source:
            print("\nListening... (speak now)")
            self.speak("I'm listening...")
            try:
                audio = self.recognizer.listen(source, timeout=5)
                print("Processing speech...")
                text = self.recognizer.recognize_google(audio)
                print(f"\nYou said: {text}")
                return text
            except sr.WaitTimeoutError:
                print("\nNo speech detected within timeout.")
                self.speak("I didn't hear anything. Please try again.")
                return ""
            except sr.UnknownValueError:
                print("\nCould not understand the audio")
                self.speak("I couldn't understand that. Please try again.")
                return ""
            except sr.RequestError as e:
                print(f"\nCould not request results; {e}")
                self.speak("I'm having trouble with speech recognition. Please try again.")
                return ""

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
            answer = "The required information is not available in the internal knowledge base."
            self.speak(answer)
            return answer

        # If Ollama is available, call it; otherwise use HF pipeline
        if self.use_ollama:
            try:
                gen = ollama.generate(model=self.ollama_model, prompt=prompt)
                # Ollama's response text can be in .response or str(gen)
                text = getattr(gen, "response", None)
                if text is None:
                    text = str(gen)
                answer_text = self._extract_answer_text(text)
                # Speak only the concise answer, print full output (including sources)
                if answer_text:
                    self.speak(answer_text)
                else:
                    self.speak(text)
                return text
            except Exception as e:
                print(f"Ollama error: {e}")
                error_msg = "I encountered an error while generating the answer with Ollama."
                self.speak(error_msg)
                return error_msg
        else:
            try:
                response = self.llm.predict(prompt)
                if not response.strip():
                    error_msg = "I encountered an error while generating the answer."
                    self.speak(error_msg)
                    return error_msg
                answer_text = self._extract_answer_text(response)
                if answer_text:
                    self.speak(answer_text)
                else:
                    self.speak(response)
                return response
            except Exception as e:
                print(f"Language model error: {e}")
                error_msg = "I encountered an error while generating the answer."
                self.speak(error_msg)
                return error_msg


def main():
    yoad = YoAdRAG()
    print("YoAd is ready. Speak a question or type 'exit' to quit.")
    print("Type 't' to use text input or 'v' to use voice input.")
    while True:
        input_mode = input("\nInput mode (t/v): ").strip().lower()
        if input_mode in ("exit", "quit"):
            print("Goodbye.")
            break
        
        if input_mode == 'v':
            q = yoad.listen()
            if not q:  # If speech recognition failed
                continue
        else:  # Default to text input
            q = input("You: ").strip()
            if q.lower() in ("exit", "quit"):
                print("Goodbye.")
                break
        
        print("\nYoAd is thinking...")
        ans = yoad.answer(q)
        print("\nYoAd:", ans)


if __name__ == "__main__":
    main()