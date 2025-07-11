# processing.py
import io
import json
import tempfile
import time
import os
from google.cloud import storage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import FAISS

from . import config

def get_current_pdf_state(storage_client, bucket):
    """Obtiene el estado actual de los PDFs en GCS (nombre y fecha de modificación)."""
    pdf_state = {}
    blobs = storage_client.list_blobs(bucket, prefix=config.ROOT_GCS_FOLDER)
    for blob in blobs:
        if blob.name.lower().endswith(".pdf") and not blob.name.startswith(config.IMAGE_FOLDER_PREFIX):
            pdf_state[blob.name] = blob.updated.isoformat()
    return pdf_state

def get_last_processed_state(bucket):
    """Lee el manifiesto desde GCS para saber qué se procesó la última vez."""
    try:
        manifest_path = f"{config.FAISS_INDEX_GCS_FOLDER}{config.PROCESSED_FILES_MANIFEST}"
        blob = bucket.blob(manifest_path)
        if not blob.exists():
            return {} # No hay manifiesto, es la primera vez
        manifest_data = blob.download_as_string()
        return json.loads(manifest_data)
    except Exception as e:
        print(f"No se pudo leer el manifiesto anterior: {e}")
        return {} # Tratar como si fuera la primera vez

def process_and_upload_index(st_status_container):
    """
    Función principal de procesamiento. Compara estados, procesa PDFs si es necesario,
    y sube el nuevo índice y manifiesto a GCS.
    """
    storage_client = storage.Client(project=config.PROJECT_ID)
    bucket = storage_client.bucket(config.BUCKET_NAME)

    st_status_container.update(label="Paso 1/5: Verificando cambios en los PDFs de GCS...", state="running")
    current_state = get_current_pdf_state(storage_client, bucket)
    last_state = get_last_processed_state(bucket)

    if not current_state:
        st_status_container.update(label="No se encontraron PDFs en la ruta especificada. Proceso detenido.", state="error", expanded=True)
        return False, "No se encontraron PDFs."

    # Si no hay cambios, no hacemos nada.
    if current_state == last_state:
        st_status_container.update(label="¡No hay cambios! Los documentos ya están actualizados.", state="complete", expanded=False)
        return True, "El índice ya está actualizado."

    st_status_container.update(label=f"Paso 2/5: Se detectaron cambios. Procesando {len(current_state)} PDFs...", state="running")

    # --- Carga y división de documentos ---
    all_docs = []
    pdf_blobs_to_process = [bucket.get_blob(name) for name in current_state.keys()]
    
    for i, blob in enumerate(pdf_blobs_to_process):
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_pdf:
                blob.download_to_filename(temp_pdf.name)
                loader = PyMuPDFLoader(temp_pdf.name)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = blob.name
                all_docs.extend(docs)
            # Actualiza el estado en la UI
            st_status_container.update(label=f"Paso 2/5: Cargando PDFs... ({i+1}/{len(pdf_blobs_to_process)}) - {os.path.basename(blob.name)}", state="running")
        except Exception as e:
            print(f"Error procesando {blob.name}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = text_splitter.split_documents(all_docs)

    st_status_container.update(label=f"Paso 3/5: Creando embeddings para {len(chunks)} fragmentos de texto...", state="running")
    embeddings = VertexAIEmbeddings(model_name=config.EMBEDDING_MODEL_NAME, project=config.PROJECT_ID)
    
    start_time = time.time()
    vector_store = FAISS.from_documents(chunks, embeddings)
    end_time = time.time()
    print(f"Índice FAISS creado en {end_time - start_time:.2f} segundos.")

    st_status_container.update(label="Paso 4/5: Guardando y subiendo el nuevo índice a GCS...", state="running")
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store.save_local(temp_dir)
        for filename in ["index.faiss", "index.pkl"]:
            gcs_path = f"{config.FAISS_INDEX_GCS_FOLDER}{filename}"
            blob_to_upload = bucket.blob(gcs_path)
            blob_to_upload.upload_from_filename(os.path.join(temp_dir, filename))

    # --- Guardar el nuevo manifiesto ---
    manifest_blob = bucket.blob(f"{config.FAISS_INDEX_GCS_FOLDER}{config.PROCESSED_FILES_MANIFEST}")
    manifest_blob.upload_from_string(json.dumps(current_state, indent=2), content_type="application/json")
    
    st_status_container.update(label="Paso 5/5: ¡Proceso completado con éxito!", state="complete", expanded=False)
    return True, "El índice se ha actualizado correctamente."