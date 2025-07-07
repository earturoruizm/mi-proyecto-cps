# 1. Usar una imagen base oficial de Python
FROM python:3.9-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Instalar librerías del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# 4. Copiar el archivo de requerimientos e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Descargar los modelos de NLP durante la construcción del contenedor
RUN python -m spacy download es_core_news_sm
RUN python -c "import nltk; nltk.download('stopwords')"

# --- LÍNEA NUEVA ---
COPY download_model.py .
RUN python download_model.py
# --- FIN LÍNEA NUEVA ---

# 6. Copiar el resto del código de tu aplicación
COPY . .

# 7. Exponer el puerto en el que correrá la aplicación
EXPOSE 8080

# 8. Comando para ejecutar la aplicación usando un servidor de producción
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "900", "main:app"]
