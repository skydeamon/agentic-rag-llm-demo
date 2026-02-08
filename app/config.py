
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    provider: str = os.getenv('PROVIDER', 'openai')
    model_name: str = os.getenv('MODEL_NAME', 'gpt-4o-mini')
    embeddings_provider: str = os.getenv('EMBEDDINGS_PROVIDER', 'sentence-transformers')
    embeddings_model: str = os.getenv('EMBEDDINGS_MODEL', 'all-MiniLM-L6-v2')
    vector_store_path: str = os.getenv('VECTOR_STORE_PATH', '.vector_store/faiss')
    app_host: str = os.getenv('APP_HOST', '0.0.0.0')
    app_port: int = int(os.getenv('APP_PORT', '8000'))

settings = Settings()
