import os
import textwrap
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import speech_recognition as sr
import pyttsx3
import openai
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

# External LLM config (optional)
EXTERNAL_LLM_PROVIDER = os.getenv("EXTERNAL_LLM_PROVIDER")
EXTERNAL_MODE = os.getenv("EXTERNAL_MODE", "kb_then_external")


class YoAdRAG:
    def __init__(self, index_name: str = INDEX_NAME, top_k: int = 5):
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
You are YoAd, a polite, precise, and academically oriented AI tutor.
Your primary goal is to help the user understand, compare, and test academic concepts clearly and effectively.

Response Guidelines

Concept Explanations:
- Explain concepts in simple, beginner-friendly language.
- Use short paragraphs and clear examples to illustrate key ideas.
- When helpful, include step-by-step reasoning or analogies for better understanding.

Comparative Questions:
- Present differences or similarities in a structured format (e.g., table or bullet points).
- Focus on clarity, core distinctions, and practical relevance.

Multiple Choice Questions (MCQs):
- Generate exactly 10 well-balanced multiple choice questions based on the provided context or topic.
- Each question must be numbered from 1 to 10.
- Each question must have 4 options (a, b, c, d).
- After each question, clearly mark the correct answer (e.g., "Correct answer: b").
- Do not stop after 1 question—always list all 10.
- Include a mix of easy, moderate, and challenging questions that test understanding rather than memorization.

General Instructions:
- Be concise, factual, and academically neutral.
- Include examples or short scenarios to demonstrate understanding.

At the end of every response, include:
Sources: [1] [2] ... (listing the chunk numbers used).

If the context does NOT contain relevant information, respond exactly with:
“The required information is not available in the internal knowledge base.”

CONTEXT:
{context}

QUESTION: {question}

OUTPUT FORMAT (must follow):
Answer:
<your response following the guidelines above>

Sources:
[1] [2] ...
""")

    def speak(self, text: str) -> None:
        """Speak the given text using text-to-speech."""
        self.engine.say(text)
        self.engine.runAndWait()

    def _extract_answer_text(self, llm_text: str) -> str:
        """Extract the 'Answer' section from the model output.

        For MCQ/quiz queries, return the full answer block after 'Answer:'.
        For other queries, fallback to previous logic.
        """
        if not llm_text:
            return ""
        txt = llm_text
        lower = txt.lower()
        ans_idx = lower.find('answer:')
        src_idx = lower.find('\nsources:')
        # If MCQ/quiz detected in the answer, return everything after 'Answer:'
        if ans_idx != -1:
            answer = txt[ans_idx + len('answer:'):]
            # If 'Sources:' exists, cut at that point
            if src_idx != -1 and src_idx > ans_idx:
                answer = txt[ans_idx + len('answer:'):src_idx]
            # If the answer contains at least 2 MCQ numbers, return as is
            if any(x in answer.lower() for x in ["1.", "2.", "3.", "4.", "5."]):
                return answer.strip()
            # Otherwise, fallback to previous logic
            return answer.strip()
        # fallback: try to split by two newlines then take first paragraph
        parts = txt.split('\n\n')
        return parts[0].strip() if parts else txt.strip()

    def _call_external(self, prompt_text: str) -> str:
        """Call external LLM provider (supports OpenAI and OpenRouter) and return text or None on failure."""
        provider = os.getenv("EXTERNAL_LLM_PROVIDER", "").lower()
        if not provider:
            return None
            
        try:
            import openai
        except Exception as e:
            print(f"OpenAI python package not installed: {e}")
            return None
            
        if provider == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                print("External LLM configured as OpenAI but OPENAI_API_KEY is missing in .env")
                return None
            try:
                openai.api_key = key
                model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "512")),
                    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
                )
                text = resp["choices"][0]["message"]["content"].strip()
                return text
            except Exception as e:
                print(f"External LLM (OpenAI) error: {e}")
                return None
                
        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            model = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
            
            if not api_key:
                print("OpenRouter API key is missing in .env")
                return None
                
            try:
                openai.api_key = api_key
                openai.api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
                
                # Required for OpenRouter
                headers = {
                    "HTTP-Referer": "http://localhost:3000",  # Your site's URL
                    "X-Title": "YoAd Education AI Assistant"  # Your app's name
                }
                
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt_text}],
                    headers=headers,
                    max_tokens=int(os.getenv("OPENROUTER_MAX_TOKENS", "512")),
                    temperature=float(os.getenv("OPENROUTER_TEMPERATURE", "0.2")),
                )
                text = resp["choices"][0]["message"]["content"].strip()
                return text
            except Exception as e:
                print(f"External LLM (OpenRouter) error: {e}")
                return None
                
        else:
            print(f"External LLM provider '{provider}' is not supported. Use 'openai' or 'openrouter'.")
            return None

    def listen(self) -> str:
        """Listen for voice input and return the recognized text."""
        with sr.Microphone() as source:
            print("\nAdjusting for ambient noise... Please wait...")
            # Calibrate for ambient noise (longer duration helps in noisy environments)
            try:
                self.recognizer.dynamic_energy_threshold = True
                # Use a slightly longer ambient calibration when needed
                self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
            except Exception:
                # If calibration fails, continue with defaults
                pass

            # Tweak thresholds to avoid early cutoff
            # Pause threshold: seconds of silence that will register as end of phrase
            self.recognizer.pause_threshold = 0.9
            # How long to wait for a phrase to start (None = wait indefinitely)
            listen_timeout = None
            # Do not forcibly cut off phrases unless very long; keep phrase_time_limit None
            phrase_time_limit = None

            print("Listening... (speak now)")
            self.speak("I'm listening...")
            try:
                audio = self.recognizer.listen(source, timeout=listen_timeout, phrase_time_limit=phrase_time_limit)
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


    def answer(self, question: str, speak: bool = True) -> str:
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

        # For MCQ/quiz queries, always call external LLM and remove fallback phrase from prompt
        if any(x in question.lower() for x in ["mcq", "quiz", "multiple choice"]):
            # Remove fallback phrase from prompt for MCQ/quiz
            prompt_template_mcq = self.prompt_template.replace(
                '\nIf the context does NOT contain relevant information, respond exactly with:\n“The required information is not available in the internal knowledge base.”',
                ''
            )
            prompt = prompt_template_mcq.format(context=context, question=question)
            provider = os.getenv("EXTERNAL_LLM_PROVIDER", "openai").lower()
            # Set high max tokens for MCQ/quiz
            orig_max_tokens = os.getenv("OPENAI_MAX_TOKENS", "512")
            os.environ["OPENAI_MAX_TOKENS"] = "1500"
            orig_or_max_tokens = os.getenv("OPENROUTER_MAX_TOKENS", "512")
            os.environ["OPENROUTER_MAX_TOKENS"] = "1500"
            ext_resp = self._call_external(prompt)
            os.environ["OPENAI_MAX_TOKENS"] = orig_max_tokens
            os.environ["OPENROUTER_MAX_TOKENS"] = orig_or_max_tokens
            if ext_resp:
                if speak:
                    self.speak(ext_resp)
                return ext_resp

        prompt = self.prompt_template.format(context=context, question=question)

        # External LLM decision logic (default)
        ext_provider = os.getenv("EXTERNAL_LLM_PROVIDER")
        ext_mode = os.getenv("EXTERNAL_MODE", "kb_then_external")

        if ext_provider:
            mode = ext_mode.lower()
            if mode == "kb_then_external" and not context.strip():
                ext_resp = self._call_external(f"QUESTION: {question}")
                if ext_resp:
                    if speak:
                        self.speak(ext_resp)
                    return ext_resp
            elif mode in ("combine_then_external", "external_always"):
                ext_prompt = prompt
                ext_resp = self._call_external(ext_prompt)
                if ext_resp:
                    if speak:
                        self.speak(ext_resp)
                    return ext_resp

        # If context is empty, short-circuit to ensure LLM returns the required message
        if not context.strip():
            answer = "The required information is not available in the internal knowledge base."
            if speak:
                self.speak(answer)
            return answer

        # If Ollama is available, call it; otherwise use HF pipeline
        if self.use_ollama:
            try:
                # Try to use CPU if CUDA fails
                try:
                    gen = ollama.generate(model=self.ollama_model, prompt=prompt)
                except Exception as cuda_error:
                    if "CUDA" in str(cuda_error):
                        print("CUDA error detected, falling back to CPU...")
                        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA
                        gen = ollama.generate(model=self.ollama_model, prompt=prompt)
                    else:
                        raise cuda_error
                
                # Ollama's response text can be in .response or str(gen)
                text = getattr(gen, "response", None)
                if text is None:
                    text = str(gen)
                answer_text = self._extract_answer_text(text)
                # Speak only the concise answer, print full output (including sources)
                if speak:
                    # If speaking is enabled, speak only the concise answer when available
                    if answer_text:
                        self.speak(answer_text)
                    else:
                        self.speak(text)
                return text
            except Exception as e:
                print(f"Ollama error: {e}")
                print("Falling back to HuggingFace pipeline...")
                self.use_ollama = False
                # Initialize HF pipeline as fallback
                try:
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
                    return self.answer(question, speak)  # Retry with HF pipeline
                except Exception as hf_error:
                    print(f"HuggingFace pipeline error: {hf_error}")
                    error_msg = "I encountered an error. Please check if OpenAI is configured in .env for fallback."
                    if speak:
                        self.speak(error_msg)
                    return error_msg
        else:
            try:
                response = self.llm.predict(prompt)
                if not response.strip():
                    error_msg = "I encountered an error while generating the answer."
                    if speak:
                        self.speak(error_msg)
                    return error_msg
                answer_text = self._extract_answer_text(response)
                if speak:
                    if answer_text:
                        self.speak(answer_text)
                    else:
                        self.speak(response)
                return response
            except Exception as e:
                print(f"Language model error: {e}")
                error_msg = "I encountered an error while generating the answer."
                if speak:
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
            # Voice mode: listen, then get model output but don't let answer() speak.
            q = yoad.listen()
            if not q:  # If speech recognition failed
                continue
            print("\nYoAd is thinking...")
            ans = yoad.answer(q, speak=False)
            # Extract only the concise 'Answer' portion (remove Sources)
            display = yoad._extract_answer_text(ans)
            print("\nYoAd:", display)
            # Speak the concise answer
            yoad.speak(display)
        else:
            # Text mode: get answer but do NOT speak it
            q = input("You: ").strip()
            if q.lower() in ("exit", "quit"):
                print("Goodbye.")
                break
            print("\nYoAd is thinking...")
            ans = yoad.answer(q, speak=False)
            # Print only the concise 'Answer' portion (omit Sources)
            display = yoad._extract_answer_text(ans)
            print("\nYoAd:", display)


if __name__ == "__main__":
    main()