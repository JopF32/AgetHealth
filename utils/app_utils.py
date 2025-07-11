# utils/app_utils.py
import streamlit as st
import tempfile
import os
from datetime import timedelta
from google.cloud import storage

# Importaciones de LangChain
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Importamos nuestra configuración centralizada
from . import config
from . import gcs_tools
from . import agent_logic
def check_index_exists():
    """Comprueba si el archivo principal del índice (index.faiss) existe en GCS."""
    try:
        storage_client = storage.Client(project=config.PROJECT_ID)
        bucket = storage_client.bucket(config.BUCKET_NAME)
        index_blob = bucket.blob(f"{config.FAISS_INDEX_GCS_FOLDER}index.faiss")
        return index_blob.exists()
    except Exception as e:
        print(f"Error al verificar la existencia del índice: {e}")
        return False

@st.cache_resource
def load_vector_store_from_gcs(_embeddings_model):
    """
    Descarga los archivos del índice FAISS de GCS a un directorio temporal
    y los carga en memoria.
    """
    if not check_index_exists():
        st.error("El índice RAG no se encuentra en GCS. Por favor, procesa los PDFs primero.")
        return None
        
    try:
        storage_client = storage.Client(project=config.PROJECT_ID)
        bucket = storage_client.bucket(config.BUCKET_NAME)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for filename in ["index.faiss", "index.pkl"]:
                blob = bucket.blob(f"{config.FAISS_INDEX_GCS_FOLDER}{filename}")
                blob.download_to_filename(os.path.join(temp_dir, filename))
            
            vector_store = FAISS.load_local(
                temp_dir, _embeddings_model, allow_dangerous_deserialization=True
            )
        return vector_store
    except Exception as e:
        st.error(f"Error crítico al cargar el índice vectorial desde GCS: {e}")
        return None

@st.cache_resource
def load_rag_chain():
    """
    Carga y construye la cadena RAG completa.
    Esta función es el núcleo de la búsqueda de información.
    """
    print("Iniciando la carga de la cadena RAG...")
    
    # 1. Cargar el modelo de embeddings
    embeddings = VertexAIEmbeddings(**config.EMBEDDING_MODEL_CONFIG)
    
    # 2. Cargar la base de datos de vectores (índice FAISS)
    vector_store = load_vector_store_from_gcs(embeddings)
    
    # 3. Comprobación de seguridad: si el vector_store no se cargó, no podemos continuar.
    if not vector_store:
        print("Fallo al cargar Vector Store. La cadena RAG no se puede construir.")
        return None 
    
    print("Vector Store cargado. Construyendo el resto de la cadena RAG...")


    retriever = vector_store.as_retriever(search_kwargs={'k': 5})
    
    # 5. Crear el modelo de lenguaje para la respuesta
    llm = ChatVertexAI(**config.RAG_RESPONSE_LLM_CONFIG)
    
    # 6. Definir la plantilla del prompt
    template = """
    Eres un asistente experto. Responde la PREGUNTA basándote únicamente en el siguiente CONTEXTO.
    Si la respuesta no está en el CONTEXTO, di "No he encontrado información sobre eso en los documentos."
    Cita la fuente del documento si es posible (ej: 'Según el documento X.pdf...').
    CONTEXTO: {context}
    PREGUNTA: {question}
    RESPUESTA:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    print("Cadena RAG construida exitosamente.")
    
    return rag_chain

def execute_file_search_tool(details):
    """
    Ejecuta la búsqueda de archivos y formatea la respuesta.
    Ahora utiliza directamente la URL pública devuelta por gcs_tools.find_file_in_gcs.
    """
    keywords = details.get("file_keywords")
    if not keywords:
        return {"type": "error", "content": "El asistente no pudo identificar qué archivo buscas. Por favor, sé más específico (ej: 'CPU.jpeg')."}
    
    try:
        # find_file_in_gcs ahora devuelve un diccionario con 'url' o una lista de diccionarios, o None
        files_found = gcs_tools.find_file_in_gcs(keywords) 
    except Exception as e:
        return {"type": "error", "content": f"Hubo un problema técnico al buscar archivos: {e}"}

    if not files_found:
        return {"type": "message", "content": f"Lo siento, no encontré ningún archivo que coincida con '{keywords}'."}

    # Si find_file_in_gcs devuelve un solo archivo (diccionario)
    if isinstance(files_found, dict):
        file = files_found
        # Se asume que file['url'] ya contiene la URL pública
        if any(file["name"].lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
            return {"type": "image", "content": file["url"], "caption": file["name"]}
        else:
            return {"type": "link", "content": f"He encontrado el archivo: [{file['name']}]({file['url']})"}
    
    # Si find_file_in_gcs devuelve múltiples archivos (lista de diccionarios)
    elif isinstance(files_found, list):
        links = []
        for f in files_found:
            # Cada 'f' ya es un diccionario con 'name', 'path', y 'url'
            links.append(f"- [{f['name']}]({f['url']})")
        
        response_message = f"He encontrado {len(links)} archivos que coinciden con '{keywords}'. Aquí los tienes:\n\n" + "\n".join(links)
        return {"type": "message", "content": response_message}
    else:
        return {"type": "error", "content": "Formato de respuesta inesperado de la herramienta de búsqueda de archivos."}


def execute_list_files_in_folder_tool(details):
    """
    Ejecuta la herramienta para listar archivos dentro de una carpeta específica.
    Ahora utiliza directamente la URL pública devuelta por gcs_tools.list_files_in_specific_folder.
    """
    folder_name = details.get("folder_name")
    if not folder_name:
        return {"type": "error", "content": "El asistente no pudo identificar el nombre de la carpeta para listar archivos."}

    try:
        files = gcs_tools.list_files_in_specific_folder(folder_name)
    except Exception as e:
        return {"type": "error", "content": f"Hubo un problema técnico al listar archivos en '{folder_name}': {e}"}

    if not files:
        return {"type": "message", "content": f"Lo siento, no encontré ningún archivo en la carpeta '{folder_name}'."}

    # Generar una lista de enlaces usando las URLs públicas que ya vienen en 'files'
    links = []
    for f in files:
        # 'f' ya es un diccionario con 'name', 'path', y 'url'
        links.append(f"- [{f['name']}]({f['url']})")

    response_message = f"Aquí tienes los archivos que encontré en la categoría '{folder_name}':\n\n" + "\n".join(links)
    return {"type": "message", "content": response_message}

# Cargar la cadena RAG solo una vez
rag_chain = load_rag_chain()

if rag_chain is None:
    st.warning("El asistente no está completamente configurado (RAG chain no cargada). Por favor, revisa la configuración y los índices.")
else:
    st.success("Asistente RAG listo para responder preguntas y buscar archivos.")

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes del historial al recargar la app
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["type"] == "text":
            st.markdown(message["content"])
        elif message["type"] == "image":
            st.image(message["content"], caption=message["caption"])
        elif message["type"] == "link":
            st.markdown(message["content"])
        elif message["type"] == "error":
            st.error(message["content"])
        elif message["type"] == "warning":
            st.warning(message["content"])
        else: # Default para cualquier otro tipo desconocido
            st.markdown(message["content"])


