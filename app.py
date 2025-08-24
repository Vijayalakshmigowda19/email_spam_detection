import streamlit as st
import pickle
import pandas as pd

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Email Spam Detector")

# --- Single Email Prediction ---
st.header("🔍 Test a Single Email")
user_input = st.text_area("Enter the email text:")

if st.button("Predict"):
    if user_input.strip() != "":
        email_vec = vectorizer.transform([user_input])
        prediction = model.predict(email_vec)[0]
        st.write("### Prediction: ", "🚨 Spam" if prediction == 1 else "✅ Ham")
    else:
        st.warning("Please enter some text to analyze.")

# --- Bulk Email Prediction ---
st.header("📂 Bulk Email Testing (Upload CSV)")
uploaded_file = st.file_uploader("Upload a CSV file with a column named 'Message'", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "Message" not in df.columns:
        st.error("❌ CSV must contain a column named 'Message'")
    else:
        st.write("📄 Uploaded Data Sample:")
        st.write(df.head())

        # Vectorize and predict
        emails_vec = vectorizer.transform(df["Message"].astype(str))
        df["Prediction"] = model.predict(emails_vec)
        df["Prediction"] = df["Prediction"].map({0: "✅ Ham", 1: "🚨 Spam"})

        st.write("✅ Results:")
        st.write(df)

        # Download option
        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_out,
            file_name="spam_predictions.csv",
            mime="text/csv"
        )
