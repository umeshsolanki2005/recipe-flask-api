from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_cors import cross_origin
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app)

# Load and preprocess data
df = pd.read_csv('recipes.csv')
df = df.dropna(subset=['Title', 'Cleaned_Ingredients', 'Instructions', 'Image_Name'])  # remove incomplete records

# Convert Cleaned_Ingredients (stored as strings) into text
df['joined_ingredients'] = df['Cleaned_Ingredients'].apply(lambda x: ' '.join(eval(x)))

# Initialize TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['joined_ingredients'])

# ✅ Helper function to generate full image URL with .jpg
def get_image_url(image_name):
    if image_name:
        clean_name = image_name.split('.')[0]  # remove any extension
        return f"https://recipe-flask-api.onrender.com/static/images/{clean_name}.jpg"
    return None

# ✅ Home route
@app.route('/')
def home():
    return "✅ Recipe API is live!"

# ✅ Recommendation route
@app.route('/recommend', methods=['POST'])
@cross_origin()
def recommend():
    data = request.json
    user_input = data.get('ingredients', '')

    # Transform user input and calculate similarity
    input_vec = tfidf.transform([user_input])
    similarity = cosine_similarity(input_vec, tfidf_matrix)
    top_idx = similarity[0].argsort()[-5:][::-1]

    # Prepare response with image_url
    results = []
    for idx in top_idx:
        row = df.iloc[idx]
        results.append({
            "title": row['Title'],
            "joined_ingredients": row['joined_ingredients'],
            "instructions": row['Instructions'],
            "image_url": get_image_url(row['Image_Name'])
          # ✅ updated here
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
