from flask import Flask, render_template, request, jsonify
import pickle
import re
import string
from nltk.corpus import stopwords

# Load the model and vectorizer
def load_model():
    with open("spam_classifier.pkl", "rb") as model_file:
        loaded_model = pickle.load(model_file)
    with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)
    return loaded_model, loaded_vectorizer

model, vectorizer = load_model()

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub("\d+", "", text)  # Remove numbers
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Function for new predictions
def predict_spam(text):
    processed_text = clean_text(text)
    text_vectorized = vectorizer.transform([processed_text])
    prediction = model.predict(text_vectorized)
    spam_probability = model.predict_proba(text_vectorized)[0][1]
    spam_percentage = round(spam_probability * 100, 2)  # Convert to percentage


    if prediction[0] == 1:
        return "Spam", spam_percentage
    else:
        return "Not Spam", spam_percentage

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    result, probability = predict_spam(message)
    print("result: ", result)
    print("probability", probability)
    return jsonify({'prediction': result, 'probability': probability})

if __name__ == '__main__':
    app.run(debug=True)


if __name__ == '__main__':
    app.run(debug=True)