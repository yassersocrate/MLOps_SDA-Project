import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Charger les modèles à partir des fichiers .pkl sauvegardés
rf_model = joblib.load("models/random_forest_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")

# Définir les colonnes de caractéristiques pour le formulaire d'entrée
feature_columns = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
]


# Charger et pré-traiter les données d'entrée
def preprocess_input_data(input_data):
    # Convertir les données d'entrée en DataFrame
    df = pd.DataFrame([input_data], columns=feature_columns)

    # Appliquer la normalisation (en utilisant StandardScaler)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    return df_scaled


# Interface utilisateur de l'application Streamlit
st.title("Application de Prédiction de Défaut de Prêt")
st.write(
    "Entrez les informations du client pour prédire la probabilité de défaut de prêt."
)

# Formulaire d'entrée pour les détails du client
input_data = []
input_data.append(
    st.number_input("Lignes de crédit en cours", min_value=0, max_value=100, value=10)
)
input_data.append(
    st.number_input(
        "Montant du prêt en cours", min_value=0, max_value=1000000, value=50000
    )
)
input_data.append(
    st.number_input(
        "Dette totale en cours", min_value=0, max_value=1000000, value=100000
    )
)
input_data.append(
    st.number_input("Revenu", min_value=0, max_value=1000000, value=50000)
)
input_data.append(
    st.number_input("Années d'emploi", min_value=0, max_value=50, value=5)
)
input_data.append(
    st.number_input("Score FICO", min_value=300, max_value=850, value=700)
)

# Choix du modèle
model_choice = st.selectbox("Choisissez un modèle", ("Random Forest", "XGBoost"))

# Bouton prédire
if st.button("Inférer"):
    # Pré-traiter les données d'entrée
    input_data_preprocessed = preprocess_input_data(input_data)

    # Prédire les probabilités en fonction du modèle sélectionné
    if model_choice == "Random Forest":
        proba = rf_model.predict_proba(input_data_preprocessed)[:, 1][0]
    else:
        proba = xgb_model.predict_proba(input_data_preprocessed)[:, 1][0]

    # Afficher la probabilité de défaut
    st.write(f"La probabilité de défaut est : {proba:.4f}")
