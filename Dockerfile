# Utiliser une image Python comme base
FROM python:3.11

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt dans le répertoire de travail
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le contenu du répertoire courant dans le répertoire de travail du conteneur
COPY . .

# Spécifier la commande par défaut pour exécuter l'application Streamlit
CMD ["streamlit", "run", "appli.py"]
