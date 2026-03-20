# phase3_svd.py
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV

# ── Load ratings for Surprise ─────────────────────────────
ratings = pd.read_csv('ratings.csv')

reader  = Reader(rating_scale=(0.5, 5.0))
data    = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# ── Train/Test Split ──────────────────────────────────────
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# ── Tune number of factors (k) ────────────────────────────
param_grid = {'n_factors': [50, 100, 150], 'n_epochs': [20, 30]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
gs.fit(data)

print("Best RMSE:", gs.best_score['rmse'])
print("Best params:", gs.best_params['rmse'])

# ── Train best SVD model ──────────────────────────────────
best_params = gs.best_params['rmse']
svd_model   = SVD(n_factors=best_params['n_factors'],
                  n_epochs=best_params['n_epochs'])
svd_model.fit(trainset)

# ── Predict on test set ───────────────────────────────────
predictions = svd_model.test(testset)
svd_rmse    = accuracy.rmse(predictions)
print(f"\nSVD Test RMSE: {svd_rmse:.4f}")

# ── Explore Latent Features ───────────────────────────────
import numpy as np
P = svd_model.pu  # User embeddings  (n_users x k)
Q = svd_model.qi  # Item embeddings  (n_items x k)
print(f"\nUser Embeddings shape: {P.shape}")
print(f"Item Embeddings shape: {Q.shape}")
print("Dimension 1 might represent: Action vs Romance")
print("Dimension 2 might represent: Serious vs Comedy")

# Save model
import pickle
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(svd_model, f)
print("\nPhase 3 Complete ✅")