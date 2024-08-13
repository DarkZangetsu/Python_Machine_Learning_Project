import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Configuration de la page Streamlit
st.set_page_config(page_title="Analyse de l'enquête sur l'emploi en Inde", layout="wide")

# Titre de l'application
st.title("Analyse de l'enquête sur l'emploi en Inde")

# Téléchargement du fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

if uploaded_file is not None:
    # Lecture du fichier CSV téléchargé
    data = pd.read_csv(uploaded_file, index_col=0)

    # Affichage des premières lignes du dataset
    st.subheader("Aperçu des données")
    st.write(data.head())

    # Traitement et Exploration de Données
    st.header("Traitement et Exploration de Données")

    # Vérification des valeurs manquantes
    st.subheader("Valeurs manquantes")
    missing_values = data.isnull().sum()
    st.write(missing_values)

    # Vérification des doublons
    st.subheader("Doublons")
    duplicates = data.duplicated().sum()
    st.write(f"Nombre de doublons : {duplicates}")

    # Distribution des secteurs d'emploi
    st.subheader("Distribution des secteurs d'emploi")
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='Employment Sector', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Distribution des revenus idéaux
    st.subheader("Distribution des revenus idéaux")
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='IdealYearlyIncome', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Préparation des données pour l'apprentissage
    st.header("Préparation des données pour l'apprentissage")

    # Encodage des variables catégorielles
    le_dict = {}
    categorical_cols = ['Employment Sector', 'Employment Background', 'Public Dealing', 'Degree']
    for col in categorical_cols:
        le_dict[col] = LabelEncoder()
        data[col] = le_dict[col].fit_transform(data[col])

    # Séparation des features et de la target
    X = data.drop('IdealYearlyIncome', axis=1)
    y = data['IdealYearlyIncome']

    # Encodage de la variable cible
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write("Données préparées pour l'apprentissage")

    # Apprentissage
    st.header("Apprentissage")

    # Utilisation de Random Forest pour la classification
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    st.write("Modèle Random Forest entraîné")

    # Validation
    st.header("Validation")

    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calcul de l'accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy du modèle : {accuracy:.2f}")

    # Affichage du rapport de classification
    st.subheader("Rapport de classification")
    report = classification_report(y_test, y_pred, target_names=le_target.classes_)
    st.text(report)

    # Importance des features
    st.subheader("Importance des features")
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax)
    st.pyplot(fig)

    # Prédiction interactive
    st.header("Prédiction interactive")

    # Création de widgets pour l'entrée utilisateur
    employment_sector = st.selectbox("Secteur d'emploi", options=le_dict['Employment Sector'].classes_)
    employment_background = st.selectbox("Background d'emploi", options=le_dict['Employment Background'].classes_)
    public_dealing = st.selectbox("Public Dealing", options=le_dict['Public Dealing'].classes_)
    degree = st.selectbox("Diplôme", options=le_dict['Degree'].classes_)
    ideal_workdays = st.slider("Nombre idéal de jours de travail", min_value=4, max_value=7, value=5)

    # Préparation des données d'entrée pour la prédiction
    input_data = pd.DataFrame({
        'Employment Sector': [employment_sector],
        'Employment Background': [employment_background],
        'Public Dealing': [public_dealing],
        'Degree': [degree],
        'IdealNumberOfWorkdays': [ideal_workdays]
    })

    # Encodage des données d'entrée
    for col in categorical_cols:
        input_data[col] = le_dict[col].transform(input_data[col])

    # S'assurer que toutes les colonnes du modèle sont présentes dans input_data
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0  # ou une autre valeur par défaut appropriée

    # Réorganiser les colonnes pour correspondre à l'ordre du modèle
    input_data = input_data.reindex(columns=X.columns)

    # Prédiction
    if st.button("Prédire le revenu idéal"):
        prediction = model.predict(input_data)
        predicted_income = le_target.inverse_transform(prediction)[0]
        st.success(f"Le revenu annuel idéal prédit est : {predicted_income}")

else:
    st.info("Veuillez télécharger un fichier CSV pour commencer l'analyse.")