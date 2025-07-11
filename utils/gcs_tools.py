# gcs_tools.py
from google.cloud import storage
import os
# [CAMBIO] Importamos datetime y timedelta para manejar la expiración de la URL
from datetime import datetime, timedelta

from . import config

from google.auth import compute_engine
from google.auth.transport import requests as google_requests

# [NUEVO] Definimos cuánto tiempo serán válidas las URLs firmadas.
# 3600 segundos = 1 hora. Puedes ajustar este valor según tus necesidades.
SIGNED_URL_EXPIRATION_SECONDS = 3600


storage_client = storage.Client(project=config.PROJECT_ID)
try:
    # Obtenemos las credenciales del entorno de Compute Engine/Cloud Run
    credentials = compute_engine.Credentials()
    # Usamos las credenciales para obtener la cuenta de servicio asociada
    SERVICE_ACCOUNT_EMAIL = credentials.service_account_email
    print(f"[GCS_TOOL] Usando la cuenta de servicio del entorno: {SERVICE_ACCOUNT_EMAIL}")
except Exception:
    # Si esto falla, es porque no estamos en un entorno de GCP.
    # El código fallará más adelante, pero esto lo hace explícito.
    SERVICE_ACCOUNT_EMAIL = None
    print("[GCS_TOOL] ADVERTENCIA: No se pudo obtener la cuenta de servicio del entorno. La firma de URL fallará si no se configura una clave JSON.")

def find_file_in_gcs(keywords: str):
    """
    Busca archivos en GCS y devuelve una URL firmada (temporal y segura) para el acceso.
    """
    if not keywords:
        return None 

    print(f"[GCS_TOOL] Buscando archivos con palabras clave: '{keywords}'")
    found_files = []
    bucket = storage_client.bucket(config.BUCKET_NAME)
    
    keyword_parts = keywords.lower().split()

    expiration_time = datetime.utcnow() + timedelta(seconds=SIGNED_URL_EXPIRATION_SECONDS)

    for folder_prefix in config.SEARCHABLE_FILE_FOLDERS:
        blobs = bucket.list_blobs(prefix=folder_prefix)
        for blob in blobs:
            if not blob.name.endswith('/') and all(part in blob.name.lower() for part in keyword_parts):
                
                # [CAMBIO] Generamos una URL firmada en lugar de una URL pública.
                # Usamos la versión 'v4', que es la más recomendada.
                signed_url = blob.generate_signed_url(
                    version="v4",
                    expiration=expiration_time,
                    method="GET",
                    credentials=credentials
                )
                
                found_files.append({
                    "name": os.path.basename(blob.name),
                    "path": blob.name,
                    "url": signed_url 
                })
    
    print(f"[GCS_TOOL] Encontrados {len(found_files)} archivos coincidentes.")
    
    if len(found_files) == 1:
        return found_files[0] 
    elif len(found_files) > 1:
        return found_files
    else:
        return None


def list_files_in_specific_folder(folder_name: str):
    """
    Lista archivos en GCS y devuelve sus URLs firmadas (temporales y seguras).
    """
    if not folder_name:
        print("[GCS_TOOL] folder_name es obligatorio.")
        return []

    folder_name_lower = folder_name.lower().strip()
    print(f"[GCS_TOOL] Intentando listar archivos para: '{folder_name}'")
    found_files = []
    bucket = storage_client.bucket(config.BUCKET_NAME)

    # [IMPORTANTE] Reutilizamos la lógica de expiración para las URLs.
    expiration_time = datetime.utcnow() + timedelta(seconds=SIGNED_URL_EXPIRATION_SECONDS)

    is_extension_like = folder_name_lower in ["pdf", "jpg", "jpeg", "png", "gif", "docx", "xlsx", "pptx", "txt"]
    search_bases = [p.strip().rstrip('/') + '/' for p in config.SEARCHABLE_FILE_FOLDERS]
    if not search_bases:
        print("[GCS_TOOL] config.SEARCHABLE_FILE_FOLDERS no está configurado.")
        return []

    print(f"[GCS_TOOL] Bases de búsqueda configuradas: {search_bases}")

    for base_prefix in search_bases:
        prefix_to_search = base_prefix
        # Si no es una extensión, construimos el prefijo de carpeta completo
        if not is_extension_like:
            prefix_to_search = f"{base_prefix}{folder_name_lower}/"

        print(f"[GCS_TOOL] Buscando blobs con prefijo: '{prefix_to_search}'")
        blobs_iterator = bucket.list_blobs(prefix=prefix_to_search)

        for blob in blobs_iterator:
            # Condición para agregar el archivo
            should_add = False
            # Si buscamos por extensión, comprobamos que el nombre termine con ella
            if is_extension_like:
                if blob.name.lower().endswith(f".{folder_name_lower}") and not blob.name.endswith('/'):
                    should_add = True
            # Si buscamos por carpeta, simplemente añadimos todo lo que no sea una carpeta
            else:
                if not blob.name.endswith('/'):
                    should_add = True
            
            if should_add:
                # [CAMBIO] Generamos la URL firmada para cada archivo encontrado.
                signed_url = blob.generate_signed_url(
                    version="v4",
                    expiration=expiration_time,
                    method="GET",
                    credentials=credentials
                )
                found_files.append({
                    "name": os.path.basename(blob.name),
                    "path": blob.name,
                    # [CAMBIO] Usamos la nueva URL firmada.
                    "url": signed_url
                })
    
    unique_files = {f["path"]: f for f in found_files}.values()

    print(f"[GCS_TOOL] Encontrados {len(unique_files)} archivos para la solicitud '{folder_name}'.")
    return list(unique_files)