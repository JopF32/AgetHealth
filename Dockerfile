FROM python:3.10-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia el archivo de requerimientos primero para aprovechar el cache de Docker
COPY requirements.txt requirements.txt

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt 

# Copia el resto del código de la aplicación al directorio de trabajo
COPY . .

# Variables de entorno para Streamlit (pueden ser sobrescritas en Cloud Run)

# Expone el puerto en el que Streamlit se ejecutará (coincide con ENV PORT)
EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0", "--server.headless", "true", "--server.runOnSave", "false"]
