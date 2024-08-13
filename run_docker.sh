#!/bin/bash

# Nom de l'image Docker
IMAGE_NAME="streamlit-app"

# Nom du conteneur Docker
CONTAINER_NAME="streamlit-container"

# Vérifier si l'image existe déjà
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "L'image n'existe pas. Construction de l'image Docker..."
    docker build -t $IMAGE_NAME .
else
    echo "L'image existe déjà."
fi

# Arrêt et suppression du conteneur existant s'il existe
echo "Arrêt et suppression du conteneur existant..."
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Exécution du conteneur Docker
echo "Démarrage du conteneur Docker..."
docker run -d --name $CONTAINER_NAME -p 8501:8501 -v $(pwd):/app $IMAGE_NAME

echo "L'application Streamlit est maintenant accessible à l'adresse http://localhost:8501"