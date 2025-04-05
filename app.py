# flask-api/app.py

from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and preprocess data
df = pd.read_json('recipes.json')
df['joined_ingredients'] = df['ingredients'].apply(lambda x: ' '.join(x))

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['joined_ingredients'])

joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
df.to_csv('recipes.csv', index=False)

# ✅ Home route (for testing if API is live)
@app.route('/')
def home():
    return "✅ Recipe API is live!"

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    ingredients = data.get('ingredients', '')
    input_vec = tfidf.transform([ingredients])
    similarity = cosine_similarity(input_vec, tfidf_matrix)
    top_idx = similarity[0].argsort()[-5:][::-1]
    results = df.iloc[top_idx][['id', 'joined_ingredients']].to_dict(orient='records')
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
