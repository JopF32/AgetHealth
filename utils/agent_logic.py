# agent_logic.py
# agent_logic.py
import json
import re
from langchain_google_vertexai import ChatVertexAI
from langchain.schema.output_parser import StrOutputParser

from . import config
from . import gcs_tools

routing_llm = ChatVertexAI(**config.ROUTING_LLM_CONFIG)

ROUTING_PROMPT_TEMPLATE = """
Eres un asistente de inteligencia artificial especializado en la gestión y recuperación de documentos y en la consulta de una base de conocimiento. Tu tarea es analizar la solicitud del usuario en español e identificar la **intención más precisa** y extraer los **detalles clave** necesarios para ejecutar la acción.

**Tu respuesta DEBE ser un objeto JSON válido y NADA MÁS. No incluyas explicaciones, comentarios, o texto adicional antes o después del JSON.**

### Herramientas y sus Intenciones (ELIGE SIEMPRE LA MÁS ESPECÍFICA):

1.  **`find_specific_file`**:
    * **Propósito**: Recuperar un archivo *único y específico* del almacenamiento en la nube (GCS).
    * **Criterio de Uso**: El usuario DEBE solicitar un archivo, documento, imagen, foto, plano, PDF o manual por su **nombre exacto** o con **palabras clave que lo identifiquen sin ambigüedad** (ej. "el plano de la fachada", "la factura de enero", "la imagen del CPU"). Si el usuario pide "la foto" de algo, y ese "algo" es el identificador principal, usa esta intención.
    * **Parámetros**:
        * `file_keywords`: **(STRING - OBLIGATORIO)** Las palabras clave o el nombre del archivo. No dejes esto vacío si usas esta intención.

2.  **`list_files_in_folder`**:
    * **Propósito**: Mostrar una *lista* de archivos contenidos *dentro de una carpeta o categoría* específica.
    * **Criterio de Uso**: El usuario DEBE pedir una "lista", "qué hay", "todos los archivos", o "muéstrame las [tipo de archivo] de la carpeta [nombre de carpeta/categoría]". Se requiere que mencione una **categoría o tipo de archivo (ej. "PDFs", "fotos") O un nombre de carpeta** (ej. "manuales", "contratos").
    * **Parámetros**:
        * `folder_name`: **(STRING - OBLIGATORIO)** El nombre de la carpeta o la categoría de documentos. No dejes esto vacío si usas esta intención.

3.  **`search_knowledge_base`**:
    * **Propósito**: Responder a preguntas generales, proporcionar información conceptual o procedimental, o manejar solicitudes que NO sean explícitamente para obtener o listar archivos.
    * **Criterio de Uso**: Esta es la **opción por defecto para cualquier consulta que NO encaje perfectamente** en `find_specific_file` o `list_files_in_folder`.
        * **NO usar si la solicitud claramente pide un archivo específico o una lista de archivos en una carpeta.**
        * **EJEMPLOS DE USO**: "¿Cómo se calibra X?", "Explícame sobre Y", "Información general de Z", "Tengo una pregunta sobre [tema]", "Cuál es el procedimiento para...", o solicitudes ambiguas de archivos como "dame archivos" (sin especificar qué o de dónde).
    * **Parámetros**:
        * `question`: **(STRING - OBLIGATORIO)** La pregunta original completa del usuario.

Responde ÚNICAMENTE en formato JSON. No incluyas explicaciones, notas, o texto adicional antes o después del JSON.

### Ejemplos Detallados para el Entrenamiento:

**Caso 1: `find_specific_file` (Pedir un archivo CONCRETO)**
* **Usuario**: "dame la fotografia del CPU"
    **JSON**: ```json
    {{
        "intencion": "find_specific_file",
        "detalles": {{
            "file_keywords": "fotografia del CPU"
        }}
    }}
    ```
* **Usuario**: "muéstrame la imagen CPU.jpeg"
    **JSON**: ```json
    {{
        "intencion": "find_specific_file",
        "detalles": {{
            "file_keywords": "CPU.jpeg"
        }}
    }}
    ```
* **Usuario**: "necesito el contrato de arrendamiento"
    **JSON**: ```json
    {{
        "intencion": "find_specific_file",
        "detalles": {{
            "file_keywords": "contrato de arrendamiento"
        }}
    }}
    ```
* **Usuario**: "quiero ver el PDF de especificaciones"
    **JSON**: ```json
    {{
        "intencion": "find_specific_file",
        "detalles": {{
            "file_keywords": "PDF de especificaciones"
        }}
    }}
    ```
* **Usuario**: "baja el manual de usuario del modelo XZ-100"
    **JSON**: ```json
    {{
        "intencion": "find_specific_file",
        "detalles": {{
            "file_keywords": "manual de usuario del modelo XZ-100"
        }}
    }}
    ```

**Caso 2: `list_files_in_folder` (Pedir una LISTA de archivos de una CATEGORÍA/CARPETA)**
* **Usuario**: "lista los archivos pdf"
    **JSON**: ```json
    {{
        "intencion": "list_files_in_folder",
        "detalles": {{
            "folder_name": "pdf"
        }}
    }}
    ```
* **Usuario**: "muéstrame las fotos de la carpeta imágenes"
    **JSON**: ```json
    {{
        "intencion": "list_files_in_folder",
        "detalles": {{
            "folder_name": "imágenes"
        }}
    }}
    ```
* **Usuario**: "¿Qué planos están disponibles?"
    **JSON**: ```json
    {{
        "intencion": "list_files_in_folder",
        "detalles": {{
            "folder_name": "planos"
        }}
    }}
    ```
* **Usuario**: "todos los manuales que tengas"
    **JSON**: ```json
    {{
        "intencion": "list_files_in_folder",
        "detalles": {{
            "folder_name": "manuales"
        }}
    }}
    ```
* **Usuario**: "dime los reportes que hay"
    **JSON**: ```json
    {{
        "intencion": "list_files_in_folder",
        "detalles": {{
            "folder_name": "reportes"
        }}
    }}
    ```

**Caso 3: `search_knowledge_base` (Preguntas GENERALES o NO relacionadas con archivos Específicos/Listas)**
* **Usuario**: "¿Cuál es el procedimiento de seguridad?"
    **JSON**: ```json
    {{
        "intencion": "search_knowledge_base",
        "detalles": {{
            "question": "¿Cuál es el procedimiento de seguridad?"
        }}
    }}
    ```
* **Usuario**: "Información sobre el nuevo proyecto"
    **JSON**: ```json
    {{
        "intencion": "search_knowledge_base",
        "detalles": {{
            "question": "Información sobre el nuevo proyecto"
        }}
    }}
    ```
* **Usuario**: "Tengo una duda general"
    **JSON**: ```json
    {{
        "intencion": "search_knowledge_base",
        "detalles": {{
            "question": "Tengo una duda general"
        }}
    }}
    ```
* **Usuario**: "Muéstrame todos los archivos" (demasiado general, sin categoría/nombre específico)
    **JSON**: ```json
    {{
        "intencion": "search_knowledge_base",
        "detalles": {{
            "question": "Muéstrame todos los archivos"
        }}
    }}
    ```
* **Usuario**: "Necesito ayuda"
    **JSON**: ```json
    {{
        "intencion": "search_knowledge_base",
        "detalles": {{
            "question": "Necesito ayuda"
        }}
    }}
    ```

Pregunta del usuario: "{user_query}"
Respuesta:
"""

def clean_json_string(s):
   
    s = s.strip()
    match = re.search(r'```(?:json\s*)?([\s\S]*?)```', s, re.DOTALL)
    if match:
        return match.group(1).strip()
    return s.strip() # En caso de que no haya fence, intenta limpiar igual

def get_agent_decision(user_query):
  
    chain = routing_llm | StrOutputParser()
    
    # Formatear el prompt con la consulta del usuario
    full_prompt = ROUTING_PROMPT_TEMPLATE.format(user_query=user_query)
    
    print(f"--- PROMPT ENVIADO A GEMINI ---\n{full_prompt}\n------------------------------") # Para depuración

    raw_response = chain.invoke(full_prompt)
    
    print(f"--- RESPUESTA CRUDA DE GEMINI ---\n{raw_response}\n--------------------------------") # Para depuración

    try:
        cleaned_response = clean_json_string(raw_response)
        parsed_json = json.loads(cleaned_response)
        print(f"--- JSON PARSEADO ---\n{json.dumps(parsed_json, indent=2)}\n-----------------------") # Para depuración
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"Error al decodificar JSON: {e}. Respuesta original (limpiada): '{cleaned_response}'")
        # Fallback a search_knowledge_base si el JSON es inválido
        return {"intencion": "search_knowledge_base", "detalles": {"question": user_query}}
def execute_file_search_tool(details):
    """
     Ejecuta la búsqueda de archivos y formatea la respuesta.
    """
    keywords = details.get("file_keywords")
    if not keywords:
        return {"type": "error", "content": "El asistente no pudo identificar qué archivo buscas."}
    
    try:
        files = gcs_tools.find_file_in_gcs(keywords)
    except Exception as e:
        return {"type": "error", "content": f"Hubo un problema técnico al buscar archivos: {e}"}

    if not files:
        return {"type": "message", "content": f"Lo siento, no encontré ningún archivo que coincida con '{keywords}'."}

    # Si hay un solo archivo, lo mostramos directamente.
    if len(files) == 1:
        file = files[0]
        try:
            url = gcs_tools.list_files_in_specific_folder(f['path'])
            if any(file["name"].lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                return {"type": "image", "content": url, "caption": file["name"]}
            else:
                return {"type": "link", "content": f"He encontrado el archivo: [{file['name']}]({url})"}
        except Exception as e:
            return {"type": "error", "content": f"No se pudo generar el enlace para el archivo: {e}"}

    # Si hay varios, mostramos una lista.
    else:
        try:
            links = [f"- [{f['name']}]({gcs_tools.list_files_in_specific_folder(f['path'])})" for f in files]
            response_message = f"He encontrado {len(links)} archivos que coinciden con '{keywords}'. Aquí los tienes:\n\n" + "\n".join(links)
            return {"type": "message", "content": response_message}
        except Exception as e:
            return {"type": "error", "content": f"Error al generar la lista de enlaces: {e}"}