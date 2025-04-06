# flask-api/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load and preprocess data
df = pd.read_csv('recipes.csv')
df = df.dropna(subset=['Title', 'Cleaned_Ingredients', 'Instructions', 'Image_Name'])  # remove incomplete records

# Convert Cleaned_Ingredients (stored as strings) into text
df['joined_ingredients'] = df['Cleaned_Ingredients'].apply(lambda x: ' '.join(eval(x)))

# Initialize TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['joined_ingredients'])

# ✅ Home route
@app.route('/')
def home():
    return "✅ Recipe API is live!"

# ✅ Recommendation route
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_input = data.get('ingredients', '')

    # Transform user input and calculate similarity
    input_vec = tfidf.transform([user_input])
    similarity = cosine_similarity(input_vec, tfidf_matrix)
    top_idx = similarity[0].argsort()[-5:][::-1]

    # Prepare response
    results = []
    for idx in top_idx:
        row = df.iloc[idx]
        results.append({
            "title": row['Title'],
            "joined_ingredients": row['joined_ingredients'],
            "instructions": row['Instructions'],
            "image_url": f"https://recipe-flask-api.onrender.com/static/images/{row['Image_Name']}"
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
