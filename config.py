import os
from dotenv import load_dotenv

# .env 파일이 있다면 환경 변수를 로드합니다.
# 이 파일이 애플리케이션의 다른 부분보다 먼저 임포트되면,
# os.getenv를 사용하는 모든 곳에서 .env 파일의 값을 사용할 수 있습니다.
load_dotenv()

# --- General Application Settings ---
REQUEST_TIMEOUT = 15

# --- ChromaDB Settings ---
CHROMA_PATH = "./chroma_db"

# --- Hugging Face API Settings ---
# It's recommended to use a User Access Token from Hugging Face for higher rate limits.
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_EMBEDDING_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HUGGINGFACE_CHAT_MODEL = os.getenv("HUGGINGFACE_CHAT_MODEL", "google/gemma-2-9b-it")

# --- SEC EDGAR API Settings ---
# SEC에 요청 시 제공할 User-Agent. SEC의 정책에 따라 이메일 주소를 포함하는 것이 좋습니다.
# (예: "MyCoolApp/1.0 contact@example.com")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "stock-analysis-app contact@example.com")