# Utilise une image Python légère
FROM python:3.12

# Installe libgomp1 pour prendre en charge les dépendances LightGBM
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Définit le répertoire de travail
WORKDIR /app

# Copie le fichier des dépendances
COPY requirements.txt /app/requirements.txt

# Installe les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie les autres fichiers dans le conteneur
COPY . /app

# Expose le port sur lequel FastAPI écoutera
EXPOSE 8000

# Commande pour démarrer l'application FastAPI avec Uvicorn
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]