# phase2_memory_based.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── Load Data ─────────────────────────────────────────────
utility_matrix = pd.read_csv('utility_matrix.csv', index_col='userId')

# Fill NaN with 0 for similarity computation
matrix_filled = utility_matrix.fillna(0)

# ══════════════════════════════════════════════════════════
# PART A: User-User Collaborative Filtering
# ══════════════════════════════════════════════════════════

# Compute Cosine Similarity between users
user_similarity = cosine_similarity(matrix_filled)
user_sim_df = pd.DataFrame(
    user_similarity,
    index=utility_matrix.index,
    columns=utility_matrix.index
)
print("User Similarity Matrix (5x5 sample):")
print(user_sim_df.iloc[:5, :5])

def predict_user_user(user_id, movie_title, n_neighbors=10):
    """Predict rating for a user-movie pair using User-User CF."""
    if movie_title not in utility_matrix.columns:
        return None

    # Get similarity scores for this user vs all others
    sim_scores = user_sim_df[user_id].drop(user_id)

    # Only consider users who rated this movie
    rated_users = utility_matrix[movie_title].dropna().index
    sim_scores  = sim_scores[sim_scores.index.isin(rated_users)]

    if sim_scores.empty:
        return utility_matrix[movie_title].mean()

    # Top N neighbors
    top_neighbors = sim_scores.nlargest(n_neighbors)

    # Weighted average
    ratings  = utility_matrix.loc[top_neighbors.index, movie_title]
    weights  = top_neighbors.values
    prediction = np.dot(weights, ratings) / (np.sum(np.abs(weights)) + 1e-9)
    return round(prediction, 2)

# ══════════════════════════════════════════════════════════
# PART B: Item-Item Collaborative Filtering
# ══════════════════════════════════════════════════════════

# Transpose: items as rows
item_matrix   = matrix_filled.T
item_similarity = cosine_similarity(item_matrix)
item_sim_df   = pd.DataFrame(
    item_similarity,
    index=utility_matrix.columns,
    columns=utility_matrix.columns
)
print("\nItem Similarity Matrix computed ✅")

def predict_item_item(user_id, movie_title, n_neighbors=10):
    """Predict rating for a user-movie pair using Item-Item CF."""
    if movie_title not in item_sim_df.columns:
        return None

    # Movies the user has already rated
    user_ratings = utility_matrix.loc[user_id].dropna()
    if user_ratings.empty:
        return utility_matrix[movie_title].mean()

    # Similarity of target movie to movies user rated
    sim_scores = item_sim_df[movie_title][user_ratings.index]
    top_items  = sim_scores.nlargest(n_neighbors)

    ratings    = user_ratings[top_items.index]
    weights    = top_items.values
    prediction = np.dot(weights, ratings) / (np.sum(np.abs(weights)) + 1e-9)
    return round(prediction, 2)

# ── Quick Test ────────────────────────────────────────────
sample_user  = utility_matrix.index[0]
sample_movie = utility_matrix.columns[10]

print(f"\nTest User: {sample_user}, Movie: {sample_movie}")
print(f"User-User prediction: {predict_user_user(sample_user, sample_movie)}")
print(f"Item-Item prediction: {predict_item_item(sample_user, sample_movie)}")

# Save similarity matrices
import pickle

# Save as pickle files instead of CSV (much more memory efficient)
with open('user_similarity.pkl', 'wb') as f:
    pickle.dump(user_sim_df, f)

with open('item_similarity.pkl', 'wb') as f:
    pickle.dump(item_sim_df, f)

print("\nSaved similarity matrices as .pkl files")
print("Phase 2 Complete ✅")


