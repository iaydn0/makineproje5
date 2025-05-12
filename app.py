
import streamlit as st
import numpy as np
import joblib

st.title("MPG Tahmini - Linear, Ridge, Decision Tree ve PCA+LR")

model_name = st.selectbox("Model Seçin", [
    "Linear Regression",
    "Ridge Regression",
    "PCA + Linear Regression",
    "Decision Tree Regression"
])

model_files = {
    "Linear Regression": "linear_model.pkl",
    "Ridge Regression": "ridge_model.pkl",
    "PCA + Linear Regression": "pca_lr_model.pkl",
    "Decision Tree Regression": "decision_tree_model.pkl"
}

# Girdiler
model_year = st.slider("Model Yılı (1970 - 1982)", 1970, 1982, 1976)
origin_label = st.selectbox("Üretim Bölgesi", ["ABD", "Avrupa", "Japonya"])
origin_mapping = {"ABD": 1, "Avrupa": 2, "Japonya": 3}
origin = origin_mapping[origin_label]
acceleration = st.slider("0-60 mil/saat Hızlanma Süresi", 8.0, 24.8, 15.0)

# Normalizasyon
model_year_norm = (model_year - 1970) / (1982 - 1970)
origin_norm = (origin - 1) / 2
acceleration_norm = (acceleration - 8.0) / (24.8 - 8.0)

input_data = np.array([[model_year_norm, origin_norm, acceleration_norm]])

if st.button("Tahmin Et"):
    try:
        model = joblib.load(model_files[model_name])

        if model_name == "PCA + Linear Regression":
            pca = joblib.load("pca_transform.pkl")
            input_data = pca.transform(input_data)

        prediction = model.predict(input_data)
        st.success(f"{model_name} ile Tahmini MPG: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Model yüklenemedi: {str(e)}")
