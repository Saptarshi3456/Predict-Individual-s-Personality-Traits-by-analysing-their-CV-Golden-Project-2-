# Predict-Individual-s-Personality-Traits-by-analysing-their-CV-Golden-Project-2-
Develop an AI-driven system that predicts an individual's personality traits by analyzing their Curriculum Vitae (CV) or resume.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load labeled dataset (CVs/resumes with personality traits)
data = pd.read_csv('dataset.csv')

# Preprocess text data
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['CV'])
y = data['Personality_Trait']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluate model performance
y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

# Function to predict personality traits from a new CV/resume
def predict_personality(cv_text):
    cv_vector = vectorizer.transform([cv_text])
    trait = clf.predict(cv_vector)
    return trait

# Example usage:
cv_text = "Software engineer with 5+ years of experience in Python and machine learning."
predicted_trait = predict_personality(cv_text)
print('Predicted Personality Trait:', predicted_trait)
