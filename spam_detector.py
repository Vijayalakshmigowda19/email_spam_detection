import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load dataset
data = pd.read_csv("email.csv")

print("Dataset sample:")
print(data.head())
print("\nColumns:", data.columns)

# 2. Define column names
label_col = "Category"
text_col = "Message"

# 3. Clean labels: strip spaces, lowercase
data[label_col] = data[label_col].astype(str).str.strip().str.lower()

# 4. Map ham=0, spam=1
data[label_col] = data[label_col].map({"ham": 0, "spam": 1})

# 5. Drop any rows where mapping failed (NaN labels or missing messages)
data = data.dropna(subset=[label_col, text_col])

print("\nUnique labels after cleaning:", data[label_col].unique())

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data[text_col], data[label_col], test_size=0.2, random_state=42
)

# 7. Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 8. Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 9. Predict
y_pred = model.predict(X_test_vec)

# 10. Evaluate
print("\n✅ Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 11. Test with custom email
test_email = ["Congratulations! You've won a free lottery ticket. Claim now!"]
test_vec = vectorizer.transform(test_email)
prediction = model.predict(test_vec)
print("\n📩 Test Email Prediction:", "Spam" if prediction[0] == 1 else "Ham")
while True:
    email = input("\nEnter an email (or 'quit' to stop): ")
    if email.lower() == "quit":
        break
    email_vec = vectorizer.transform([email])
    prediction = model.predict(email_vec)[0]
    print("Prediction:", "Spam" if prediction == 1 else "Ham")
import pickle

# Save model and vectorizer
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and Vectorizer saved successfully!")
