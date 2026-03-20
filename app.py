# app.py - Web Interface for CineMatch
from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load all models once when server starts
print("Loading models...")
ratings        = pd.read_csv('ratings.csv')
movies         = pd.read_csv('movies.csv')
utility_matrix = pd.read_csv('utility_matrix.csv', index_col='userId')

with open('user_similarity.pkl', 'rb') as f:
    user_sim_df = pickle.load(f)
with open('item_similarity.pkl', 'rb') as f:
    item_sim_df = pickle.load(f)
with open('svd_model.pkl', 'rb') as f:
    svd_model = pickle.load(f)
print("Models loaded ✅")

def recommend_movies(user_id, n=5, model='svd'):
    if user_id in utility_matrix.index:
        already_rated = utility_matrix.loc[user_id].dropna().index.tolist()
    else:
        already_rated = []
    unseen = [m for m in movies['title'].tolist() if m not in already_rated]
    scores = {}
    if model == 'svd':
        movie_id_map = movies.set_index('title')['movieId'].to_dict()
        for title in unseen:
            mid = movie_id_map.get(title)
            if mid:
                scores[title] = svd_model.predict(user_id, mid).est
    elif model == 'user':
        if user_id in user_sim_df.index:
            sim_scores    = user_sim_df[user_id].drop(user_id)
            top_neighbors = sim_scores.nlargest(20)
            for title in unseen:
                if title in utility_matrix.columns:
                    rated = utility_matrix.loc[top_neighbors.index, title].dropna()
                    if not rated.empty:
                        w = top_neighbors[rated.index].values
                        scores[title] = np.dot(w, rated.values) / (np.sum(np.abs(w)) + 1e-9)
    elif model == 'item':
        if user_id in utility_matrix.index:
            user_rated = utility_matrix.loc[user_id].dropna()
            for title in unseen:
                if title in item_sim_df.columns:
                    sim = item_sim_df[title][
                        [t for t in user_rated.index if t in item_sim_df.index]
                    ].dropna()
                    if not sim.empty:
                        w, r = sim.values, user_rated[sim.index].values
                        scores[title] = np.dot(w, r) / (np.sum(np.abs(w)) + 1e-9)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

def get_popular(n=5):
    popular = ratings.groupby('movieId')['rating'].count().nlargest(n).index
    return movies[movies['movieId'].isin(popular)]['title'].tolist()

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>CineMatch 🎬</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #0f0f1a; color: #fff; min-height: 100vh; }
        .header { background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460); padding: 40px 20px; text-align: center; }
        .header h1 { font-size: 3em; color: #e94560; margin-bottom: 10px; }
        .header p { color: #a8b2d8; font-size: 1.1em; }
        .container { max-width: 900px; margin: 40px auto; padding: 0 20px; }
        .card { background: #1a1a2e; border-radius: 15px; padding: 30px; margin-bottom: 25px; border: 1px solid #2a2a4a; }
        .card h2 { color: #e94560; margin-bottom: 20px; font-size: 1.4em; }
        .form-row { display: flex; gap: 15px; flex-wrap: wrap; align-items: flex-end; }
        .form-group { flex: 1; min-width: 150px; }
        label { display: block; color: #a8b2d8; margin-bottom: 8px; font-size: 0.9em; }
        input, select { width: 100%; padding: 12px; background: #0f0f1a; border: 1px solid #2a2a4a; border-radius: 8px; color: #fff; font-size: 1em; }
        input:focus, select:focus { outline: none; border-color: #e94560; }
        button { background: linear-gradient(135deg, #e94560, #c23152); color: white; border: none; padding: 12px 30px; border-radius: 8px; font-size: 1em; cursor: pointer; font-weight: bold; }
        button:hover { opacity: 0.9; transform: translateY(-1px); }
        .results { margin-top: 25px; }
        .movie-card { background: #0f0f1a; border-radius: 10px; padding: 15px 20px; margin-bottom: 12px; border-left: 4px solid #e94560; display: flex; justify-content: space-between; align-items: center; }
        .movie-title { font-size: 1em; color: #ccd6f6; }
        .movie-rank { color: #e94560; font-weight: bold; margin-right: 15px; font-size: 1.2em; }
        .stars { color: #ffd700; font-size: 0.95em; }
        .popular-item { background: #0f0f1a; border-radius: 8px; padding: 12px 18px; margin-bottom: 10px; border-left: 4px solid #ffd700; color: #ccd6f6; }
        .badge { display: inline-block; background: #e94560; color: white; padding: 3px 10px; border-radius: 20px; font-size: 0.8em; margin-left: 10px; }
        .empty { color: #a8b2d8; text-align: center; padding: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 CineMatch</h1>
        <p>AI-Powered Movie Recommendation Engine using Collaborative Filtering</p>
    </div>

    <div class="container">
        <div class="card">
            <h2>🔍 Get Recommendations</h2>
            <form method="POST" action="/recommend">
                <div class="form-row">
                    <div class="form-group">
                        <label>User ID (1 - 610)</label>
                        <input type="number" name="user_id" min="1" max="610"
                               value="{{ user_id or 1 }}" required>
                    </div>
                    <div class="form-group">
                        <label>Algorithm</label>
                        <select name="model">
                            <option value="svd"  {% if model=='svd'  %}selected{% endif %}>SVD (Best)</option>
                            <option value="user" {% if model=='user' %}selected{% endif %}>User-User CF</option>
                            <option value="item" {% if model=='item' %}selected{% endif %}>Item-Item CF</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Number of Results</label>
                        <select name="n">
                            <option value="5"  {% if n==5  %}selected{% endif %}>Top 5</option>
                            <option value="10" {% if n==10 %}selected{% endif %}>Top 10</option>
                            <option value="15" {% if n==15 %}selected{% endif %}>Top 15</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>&nbsp;</label>
                        <button type="submit">🎯 Recommend</button>
                    </div>
                </div>
            </form>

            {% if recommendations is not none %}
            <div class="results">
                <h3 style="color:#a8b2d8; margin-bottom:15px;">
                    Results for User #{{ user_id }}
                    <span class="badge">{{ model.upper() }}</span>
                </h3>
                {% if recommendations %}
                    {% for title, score in recommendations %}
                    <div class="movie-card">
                        <div>
                            <span class="movie-rank">#{{ loop.index }}</span>
                            <span class="movie-title">{{ title }}</span>
                        </div>
                        <div class="stars">⭐ {{ "%.2f"|format(score) }} / 5.00</div>
                    </div>
                    {% endfor %}
                {% else %}
                    <div class="empty">No recommendations found. Try a different algorithm.</div>
                {% endif %}
            </div>
            {% endif %}
        </div>

        <div class="card">
            <h2>🔥 Popular Movies (For New Users)</h2>
            {% for title in popular %}
            <div class="popular-item">{{ loop.index }}. {{ title }}</div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    popular = get_popular(10)
    return render_template_string(HTML, recommendations=None,
                                   popular=popular, user_id=1,
                                   model='svd', n=5)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form.get('user_id', 1))
    model   = request.form.get('model', 'svd')
    n       = int(request.form.get('n', 5))
    results = recommend_movies(user_id, n=n, model=model)
    popular = get_popular(10)
    return render_template_string(HTML, recommendations=results,
                                   popular=popular, user_id=user_id,
                                   model=model, n=n)

if __name__ == '__main__':
    print("\n🚀 Starting CineMatch Web App...")
    print("👉 Open your browser and go to: http://127.0.0.1:5000")
    # Works both locally and on Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)