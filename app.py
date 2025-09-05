import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    df = pd.read_pickle("pumpkin_seeds.pkl")
    return df

df = load_data()

st.title("🎃 Pumpkin Seeds Classification App")

st.write("Dataset Preview:")
st.dataframe(df.head())

# ---------------- Features & Target ----------------
X = df[['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
        'Convex_Area', 'Equiv_Diameter', 'Eccentricity', 'Solidity', 'Extent',
        'Roundness', 'Aspect_Ration', 'Compactness']]
y = df['Class']

# ---------------- Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Model Training ----------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ---------------- Evaluation ----------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"✅ Accuracy: **{acc:.2f}**")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# ---------------- Prediction Form ----------------
st.subheader("Try Your Own Prediction 🎯")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"Enter {col}", value=float(X[col].mean()))

if st.button("Predict Class"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"🌱 Predicted Class: **{prediction}**")
