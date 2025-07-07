from sentence_transformers import SentenceTransformer
import os

# El nombre del modelo que quieres descargar
model_name = 'hiiamsid/sentence_similarity_spanish_es'

# La carpeta donde se guardará dentro de la aplicación
output_path = './model'

# Crea la carpeta si no existe
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Descarga el modelo y lo guarda en la carpeta especificada
print(f"Descargando el modelo '{model_name}' a la carpeta '{output_path}'...")
model = SentenceTransformer(model_name)
model.save(output_path)
print("Modelo descargado y guardado exitosamente.")
