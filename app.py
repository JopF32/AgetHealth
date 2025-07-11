# app.py
import streamlit as st

# Importaciones limpias y centralizadas desde el paquete 'utils'
# Este archivo ahora solo se encarga de la interfaz y la orquestación.
from utils import agent_logic
from utils.app_utils import load_rag_chain, check_index_exists
from utils.processing import process_and_upload_index

# --- Inicialización del Estado de la Aplicación ---
# Esta sección se mantiene igual, ya que es una buena práctica.
if 'index_ready' not in st.session_state:
    # Al iniciar, comprueba si el índice ya existe en GCS para no tener que procesar
    st.session_state.index_ready = check_index_exists()

st.set_page_config(page_title="Agente Inteligente", layout="wide")
st.title("🤖 Agente Inteligente de Documentos")

# --- Barra Lateral (Sidebar) ---
# La lógica del sidebar también se mantiene, es robusta.
with st.sidebar:
    st.header("Gestión de Documentos")
    if st.button("Procesar y Actualizar PDFs", type="primary", use_container_width=True):
        with st.status("Actualizando índice de documentos...", expanded=True) as status:
            # Limpiamos el caché de recursos para forzar una recarga del índice si se vuelve a crear
            st.cache_resource.clear()
            success, message = process_and_upload_index(status)
            if success:
                st.session_state.index_ready = True
                st.success(message)
                st.rerun() # Recarga la app para reflejar el nuevo estado
            else:
                st.error(message)

    if st.session_state.index_ready:
        st.success("✅ El índice de búsqueda está listo.")
    else:
        st.warning("⚠️ El índice de búsqueda no está disponible. Púlsalo para habilitar la búsqueda de información en PDFs.")

# --- Lógica Principal del Chat ---
# La gestión del historial de mensajes es correcta.
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Manejo especial para mostrar la imagen si existe en el historial
        if message.get("type") == "image":
            st.image(message["image_url"], caption=message["caption"])
        else:
            st.markdown(message["content"])

# Entrada de texto del usuario
if query := st.chat_input("Pide información o solicita un archivo..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analizando tu solicitud..."):
            # 1. El agente decide la intención usando la lógica mejorada
            decision = agent_logic.get_agent_decision(query)
            # Este log es muy útil para depuración
            st.write(f"_(Intención detectada: {decision.get('intencion')})_")

        # 2. Ejecutar la herramienta correspondiente según la decisión del agente
        intention = decision.get("intencion")

        # CAMBIO: Usamos la nueva intención 'find_specific_file'
        if intention == 'find_specific_file':
            with st.spinner("Buscando el archivo en GCS..."):
                # CAMBIO: Usamos la nueva función de herramienta 'execute_file_search_tool'
                result = agent_logic.execute_file_search_tool(decision.get('detalles', {}))
                
                # La lógica para mostrar el resultado es robusta y se mantiene
                if result['type'] == 'error':
                    response_content = result['content']
                    st.error(response_content)
                elif result['type'] == 'message':
                    response_content = result['content']
                    st.markdown(response_content)
                elif result['type'] == 'link':
                    response_content = result['content']
                    st.markdown(response_content)
                elif result['type'] == 'image':
                    # Para las imágenes, el contenido es la URL y se muestra directamente
                    st.image(result['content'], caption=result['caption'])
                    # Guardamos la información de la imagen en el historial
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Aquí tienes la imagen: {result['caption']}",
                        "type": "image",
                        "image_url": result['content'],
                        "caption": result['caption']
                    })
                    # Salimos del `if` para no añadir un mensaje duplicado
                    

                # Para todos los tipos que no son imagen, guardamos el texto en el historial
                st.session_state.messages.append({"role": "assistant", "content": response_content})
        elif intention == 'list_files_in_folder': # NUEVA INTENCIÓN
            with st.spinner("Listando archivos en la carpeta..."):
                result = agent_logic.execute_list_files_in_folder_tool(decision.get('detalles', {}))
                if result['type'] == 'error':
                    st.error(result['content'])
                    st.session_state.messages.append({"role": "assistant", "content": result['content']})
                else:
                    st.markdown(result['content'])
                    st.session_state.messages.append({"role": "assistant", "content": result['content']})

        # CAMBIO: Usamos la nueva intención 'search_knowledge_base'
        elif intention == 'search_knowledge_base':
            if not st.session_state.index_ready:
                response = "La búsqueda de información no está disponible. Por favor, procesa los PDFs primero desde la barra lateral."
                st.warning(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                with st.spinner("Buscando en la documentación..."):
                    rag_chain = load_rag_chain()
                    # --- CORRECCIÓN: AÑADIMOS ESTA COMPROBACIÓN DE SEGURIDAD ---
                    if rag_chain is None:
                        response = "Error: No se pudo cargar la base de conocimiento (índice RAG). Esto puede ocurrir si el proceso de indexación falló. Por favor, intenta 'Procesar y Actualizar PDFs' de nuevo desde la barra lateral."
                        st.error(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        # Este código ahora solo se ejecuta si rag_chain es un objeto válido
                        question_to_ask = decision.get("detalles", {}).get("question", query)
                        response = rag_chain.invoke(question_to_ask)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
        
        else:
            response = "Lo siento, no he podido entender tu solicitud. ¿Puedes reformularla?"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            