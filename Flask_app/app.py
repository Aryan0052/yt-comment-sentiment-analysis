from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import pandas as pd
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# -------------------------------
# Preprocessing function
# -------------------------------
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower().strip()                       # Lowercase & strip spaces
        comment = re.sub(r'\n', ' ', comment)                  # Remove newlines
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)  # Remove unwanted chars
        
        # Remove stopwords except important ones
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        
        return comment
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return comment

# -------------------------------
# Load MLflow model and vectorizer
# -------------------------------
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    mlflow.set_tracking_uri("http://51.21.196.10:5000/")  # Replace with your MLflow URI
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Initialize model and vectorizer
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "1", "./tfidf_vectorizer.pkl")

# -------------------------------
# Prediction route
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    # Ensure comments is always a list
    if isinstance(comments, str):
        comments = [comments]

    try:
        # 1️⃣ Preprocess comments
        preprocessed_comments = [preprocess_comment(c) for c in comments]
        
        # 2️⃣ Transform with vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # 3️⃣ Convert sparse matrix to DataFrame with correct feature names
        df_input = pd.DataFrame(transformed_comments.toarray(), columns=vectorizer.get_feature_names_out())
        
        # 4️⃣ Predict using MLflow model
        predictions = model.predict(df_input).tolist()
        predictions = [str(pred) for pred in predictions]

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # 5️⃣ Return JSON response
    response = [{"comment": comment, "sentiment": sentiment} 
                for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

# -------------------------------
# Run Flask app
# -------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
