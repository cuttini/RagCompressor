from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Input / output di base
EBOOKS_DIR = BASE_DIR / "ebooks_seac"
MARKDOWN_DIR = BASE_DIR / "markdown_seac"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

CHUNKS_PATH = ARTIFACTS_DIR / "chunks.jsonl"

# Dataset per CLaRa
STAGE1_RAW_PATH = ARTIFACTS_DIR / "stage1_raw.jsonl"
STAGE1_VALIDATED_PATH = ARTIFACTS_DIR / "stage1_validated.jsonl"
STAGE1_2_PATH = ARTIFACTS_DIR / "stage1_2_instruction.jsonl"

# vLLM / Qwen
# vLLM / Qwen
VLLM_BASE_URL = "http://192.168.2.184:5000/v1"
VLLM_API_KEY = "EMPTY"  # vLLM lo ignora
VLLM_MODEL_NAME = "Qwen3-32B-AWQ"
CONTEXT_WINDOW_SIZE = 131072  # 128k

# Chunking
# target approssimativo di token modello, usiamo conteggio di subword Qwen
TOKENIZER_NAME = "Qwen/Qwen2.5-32B-Instruct"
MAX_TOKENS_PER_CHUNK = 800
MIN_WORDS_PER_CHUNK = 80
WINDOW_SIZE = 3  # Numero di chunk nella sliding window (es. prev + curr + next)

# Prompting
SEED = 42

# OCR Configuration
NANONETS_MODEL_ID = "nanonets/Nanonets-OCR2-3B"  # or "nanonets/Nanonets-OCR-ss"

# PDF Processing
MAX_PAGES_PER_PDF = 50  # Set to an integer to limit pages per PDF, or None for all pages
