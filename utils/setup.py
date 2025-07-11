# build_index.py
import os
import io
import tempfile
import time
from google.cloud import storage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import FAISS

# Importamos la configuración central
from . import config

def build_and_upload_index():
    """
    Lee PDFs de GCS, los procesa en memoria, crea un índice FAISS
    y sube los archivos del índice de vuelta a GCS.
    """
    print("--- Iniciando Proceso de Indexación ---")
    
    # 1. Conectar a GCS y listar los PDFs
    storage_client = storage.Client(project=config.PROJECT_ID)
    bucket = storage_client.bucket(config.BUCKET_NAME)
    
    all_blobs = list(bucket.list_blobs(prefix=config.ROOT_GCS_FOLDER))
    
    pdf_blobs = [
        blob for blob in all_blobs 
        if blob.name.lower().endswith(".pdf") and not blob.name.startswith(config.IMAGE_FOLDER_PREFIX)
    ]

    if not pdf_blobs:
        print("¡Error! No se encontraron archivos PDF en la ruta especificada.")
        return

    print(f"Encontrados {len(pdf_blobs)} archivos PDF para procesar.")

    # 2. Cargar documentos en memoria
    all_docs = []
    print("Cargando y procesando PDFs desde GCS (esto puede tardar)...")
    for blob in pdf_blobs:
        try:
            # Descargar el PDF a un stream de bytes en memoria
            pdf_bytes = blob.download_as_bytes()
            
            # PyMuPDFLoader necesita un path, así que usamos un archivo temporal
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_pdf:
                temp_pdf.write(pdf_bytes)
                temp_pdf.seek(0)
                
                loader = PyMuPDFLoader(temp_pdf.name)
                docs = loader.load()
                # Añadir la fuente (nombre del archivo) a la metadata de cada página
                for doc in docs:
                    doc.metadata["source"] = blob.name
                all_docs.extend(docs)
            print(f" - Procesado: {blob.name}")
        except Exception as e:
            print(f"  - Error procesando {blob.name}: {e}")

    # 3. Dividir documentos en fragmentos (chunks)
    print("\nDividiendo documentos en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = text_splitter.split_documents(all_docs)
    print(f"Total de fragmentos creados: {len(chunks)}")

    # 4. Crear embeddings y el índice FAISS
    print("\nCreando embeddings con Vertex AI y construyendo el índice FAISS...")
    start_time = time.time()
    embeddings = VertexAIEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME, project=config.PROJECT_ID
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    end_time = time.time()
    print(f"Índice FAISS creado en memoria en {end_time - start_time:.2f} segundos.")

    # 5. Guardar el índice en GCS
    print("\nSubiendo el índice FAISS a Google Cloud Storage...")
    # FAISS.save_local crea dos archivos: index.faiss y index.pkl
    # Los guardamos en un directorio temporal y de ahí los subimos a GCS
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store.save_local(temp_dir)
        
        # Subir index.faiss
        blob_faiss = bucket.blob(f"{config.FAISS_INDEX_GCS_FOLDER}index.faiss")
        blob_faiss.upload_from_filename(os.path.join(temp_dir, "index.faiss"))
        
        # Subir index.pkl
        blob_pkl = bucket.blob(f"{config.FAISS_INDEX_GCS_FOLDER}index.pkl")
        blob_pkl.upload_from_filename(os.path.join(temp_dir, "index.pkl"))
    
    print("--- ¡Éxito! El índice ha sido construido y guardado en GCS. ---")
    print(f"Ruta del índice en GCS: gs://{config.BUCKET_NAME}/{config.FAISS_INDEX_GCS_FOLDER}")

if __name__ == "__main__":
    build_and_upload_index()