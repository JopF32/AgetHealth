
PROJECT_ID = "kyndryl-datalake"
LOCATION = "us-central1"
GEMINI_MODEL = "gemini-2.0-flash-lite-001"
BUCKET_NAME = "becarios"

# --- Rutas GCS ---
# Carpeta raíz que contiene todos los PDFs y subcarpetas
ROOT_GCS_FOLDER = "infinitydelta/" 
# Carpeta a excluir del procesamiento
IMAGE_FOLDER_PREFIX = f"{ROOT_GCS_FOLDER}Fotos/" 
# Nombre del archivo que guardará el estado de los PDFs procesados
PROCESSED_FILES_MANIFEST = "manifest.json"
# Nuevas rutas para el índice y JSONs GLOBALES
GLOBAL_JSON_GCS_FOLDER = f"{ROOT_GCS_FOLDER}processed_json_global/"
FAISS_INDEX_GCS_FOLDER = f"{ROOT_GCS_FOLDER}faiss_index_global/"

# Modelos a usar (Vertex AI model names)
PDF_EXTRACTION_LLM = "gemini-2.0-flash-lite-001" 
EMBEDDING_MODEL_NAME = "text-embedding-004" # O "text-multilingual-embedding-002"
RAG_RESPONSE_LLM = "gemini-2.0-flash-lite-001"
ROUTING_LLM = "gemini-2.0-flash-lite-001"
ROUTING_LLM_CONFIG = {
    "model_name": "gemini-2.0-flash-lite-001",
    "project": PROJECT_ID,
    "location": LOCATION,
    "temperature": 0.1
}
EMBEDDING_MODEL_CONFIG = {
    "model_name": "text-embedding-004",
    "project": PROJECT_ID
}

RAG_RESPONSE_LLM_CONFIG = {
    "model_name": "gemini-2.0-flash-lite-001",
    "project": PROJECT_ID,
    "location": LOCATION,
    "temperature": 0.1
}
SEARCHABLE_FILE_FOLDERS = [
    f"{ROOT_GCS_FOLDER}Fotos/",
    f"{ROOT_GCS_FOLDER}Correctivo/",
    f"{ROOT_GCS_FOLDER}Preventivo/"
]
