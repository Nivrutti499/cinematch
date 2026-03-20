# phase4_evaluation_and_recommend.py
import pandas as pd
import numpy as np
import pickle
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# ── Load everything ───────────────────────────────────────
ratings        = pd.read_csv('ratings.csv')
movies         = pd.read_csv('movies.csv')
utility_matrix = pd.read_csv('utility_matrix.csv', index_col='userId')

with open('user_similarity.pkl', 'rb') as f:
    user_sim_df = pickle.load(f)

with open('item_similarity.pkl', 'rb') as f:
    item_sim_df = pickle.load(f)

with open('svd_model.pkl', 'rb') as f:
    svd_model = pickle.load(f)

print("All files loaded ✅")

# ══════════════════════════════════════════════════════════
# RMSE Comparison
# ══════════════════════════════════════════════════════════
reader     = Reader(rating_scale=(0.5, 5.0))
data       = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
_, testset = train_test_split(data, test_size=0.2, random_state=42)

svd_preds     = svd_model.test(testset)
svd_rmse      = accuracy.rmse(svd_preds, verbose=False)
global_mean   = ratings['rating'].mean()
baseline_rmse = np.sqrt(np.mean([(r - global_mean)**2
                                  for _, _, r, _, _ in svd_preds]))

print("=" * 45)
print("         RMSE COMPARISON REPORT")
print("=" * 45)
print(f"  Baseline (Global Mean): {baseline_rmse:.4f}")
print(f"  SVD Model:              {svd_rmse:.4f}  ✅ (Best)")
print("  (Lower RMSE = better predictions)")
print("=" * 45)

# ══════════════════════════════════════════════════════════
# recommend_movies() — MUST be defined before precision_at_k
# ══════════════════════════════════════════════════════════
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
        user_id_str = str(user_id)
        if user_id_str in user_sim_df.index:
            sim_scores    = user_sim_df[user_id_str].drop(user_id_str)
            top_neighbors = sim_scores.nlargest(20)
            for title in unseen:
                if title in utility_matrix.columns:
                    rated = utility_matrix.loc[
                        [int(u) for u in top_neighbors.index
                         if int(u) in utility_matrix.index], title
                    ].dropna()
                    if not rated.empty:
                        w = top_neighbors[[str(i) for i in rated.index]].values
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
                        w = sim.values
                        r = user_rated[sim.index].values
                        scores[title] = np.dot(w, r) / (np.sum(np.abs(w)) + 1e-9)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

# ══════════════════════════════════════════════════════════
# Precision@K — defined AFTER recommend_movies
# ══════════════════════════════════════════════════════════
def precision_at_k(user_id, k=10, threshold=3.5):
    if user_id not in utility_matrix.index:
        return 0.0
    liked       = utility_matrix.loc[user_id].dropna()
    liked       = liked[liked >= threshold].index.tolist()
    top_k       = recommend_movies(user_id, n=k, model='svd')
    recommended = [title for title, score in top_k]
    hits        = len([m for m in recommended if m in liked])
    return round(hits / k, 4)

# ── Precision@K Results ───────────────────────────────────
print("\n📊 Precision@K Results (K=10):")
print("-" * 40)
total      = 0
test_users = [1, 2, 3, 4, 5]
for uid in test_users:
    p = precision_at_k(uid, k=10)
    total += p
    print(f"  User #{uid}: Precision@10 = {p:.4f}")
avg_precision = total / len(test_users)
print(f"\n  Average Precision@10: {avg_precision:.4f}")
print("  (Higher = better recommendations)")

# ══════════════════════════════════════════════════════════
# Demo Output
# ══════════════════════════════════════════════════════════
demo_user = 1

print(f"\n🎬 Top 5 Recommendations for User #{demo_user} (SVD):")
svd_results = recommend_movies(demo_user, n=5, model='svd')
for i, (title, score) in enumerate(svd_results, 1):
    print(f"  {i}. {title}  (predicted rating: {score:.2f})")

print(f"\n🎬 Top 5 Recommendations for User #{demo_user} (User-User CF):")
user_results = recommend_movies(demo_user, n=5, model='user')
for i, (title, score) in enumerate(user_results, 1):
    print(f"  {i}. {title}  (predicted rating: {score:.2f})")

print(f"\n🎬 Top 5 Recommendations for User #{demo_user} (Item-Item CF):")
item_results = recommend_movies(demo_user, n=5, model='item')
for i, (title, score) in enumerate(item_results, 1):
    print(f"  {i}. {title}  (predicted rating: {score:.2f})")

# ── Cold Start ────────────────────────────────────────────
print("\n❄️  Cold Start Solution (brand new user with no history):")
popular        = ratings.groupby('movieId')['rating'].count().nlargest(5).index
popular_titles = movies[movies['movieId'].isin(popular)]['title'].tolist()
for i, t in enumerate(popular_titles, 1):
    print(f"  {i}. {t}  (popularity-based)")

print("\n" + "=" * 45)
print("   Phase 4 Complete ✅  Project Done! 🎉")
print("=" * 45)

# ── Save results for Final Report ────────────────────────
results_summary = {
    'baseline_rmse'  : baseline_rmse,
    'svd_rmse'       : svd_rmse,
    'avg_precision'  : avg_precision,
    'svd_results'    : svd_results,
    'user_results'   : user_results,
    'item_results'   : item_results,
    'popular_titles' : popular_titles,
}
with open('results_summary.pkl', 'wb') as f:
    pickle.dump(results_summary, f)
print("\nResults saved for Final Report ✅")