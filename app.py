import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Diabetes Dashboard", layout="wide")

st.markdown("""
    <style>
        .main { background-color: #0e1117; color: white; }
        .stMetric { background-color: #1c1f26; border-radius: 10px; padding: 15px; }
    </style>
""", unsafe_allow_html=True)

page = st.sidebar.radio("📌 Select Page", ["Prediction", "Dashboard"])

@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

df = load_data()

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
acc = round(model.score(X_test, y_test) * 100, 2)

if page == "Prediction":
    st.title("🩺 Diabetes Prediction")
    c1, c2 = st.columns(2)

    with c1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        glucose = st.number_input("Glucose Level", 0, 300, 120)
        blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
        skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

    with c2:
        insulin = st.number_input("Insulin", 0, 900, 79)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.number_input("Age", 1, 120, 25)

    if st.button("Predict"):
        data = np.array([[pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, bmi, dpf, age]])
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)
        probs = model.predict_proba(data_scaled)[0]

        colR1, colR2 = st.columns(2)
        with colR1:
            st.metric("🔢 Probability of Diabetes", f"{probs[1]*100:.2f}%")
            st.metric("📉 Probability of No Diabetes", f"{probs[0]*100:.2f}%")

        with colR2:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.bar(["No Diabetes", "Diabetes"], probs * 100, color=["green", "red"])
            ax.set_ylim(0, 100)
            ax.set_ylabel("Probability (%)")
            ax.set_title("Prediction Probability")
            st.pyplot(fig)

        if prediction[0] == 1:
            st.error("⚠️ The person is likely to have Diabetes.")
        else:
            st.success("✅ The person is not likely to have Diabetes.")

else:
    st.title("📊 Diabetes Dashboard")

    st.sidebar.subheader("🔍 Filter Data")
    min_age, max_age = st.sidebar.slider("Select Age Range", 0, 100, (20, 50))
    filtered_df = df[(df["Age"] >= min_age) & (df["Age"] <= max_age)]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", len(filtered_df))
    col2.metric("Diabetic", filtered_df["Outcome"].sum())
    col3.metric("Non-Diabetic", len(filtered_df)-filtered_df["Outcome"].sum())
    col4.metric("Model Accuracy", f"{acc}%")

    cA, cB, cC = st.columns(3)

    with cA:
        fig1, ax1 = plt.subplots(figsize=(3, 3))
        filtered_df["Outcome"].value_counts().plot.pie(
            autopct="%1.1f%%", colors=["red", "green"], ax=ax1
        )
        ax1.set_ylabel("")
        ax1.set_title("Diabetes Split")
        st.pyplot(fig1)

    with cB:
        fig2, ax2 = plt.subplots(figsize=(3, 3))
        sns.histplot(filtered_df["Age"], bins=10, kde=True, ax=ax2, color="cyan")
        ax2.set_title("Age Distribution")
        st.pyplot(fig2)

    with cC:
        fig3, ax3 = plt.subplots(figsize=(3, 3))
        sns.boxplot(x="Outcome", y="Glucose", data=filtered_df, ax=ax3, palette="Set2")
        ax3.set_title("Glucose by Outcome")
        st.pyplot(fig3)
