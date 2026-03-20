# phase1_data_preparation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load Data ──────────────────────────────────────────
ratings = pd.read_csv('ratings.csv')   # userId, movieId, rating, timestamp
movies  = pd.read_csv('movies.csv')    # movieId, title, genres

print("Ratings shape:", ratings.shape)
print(ratings.head())
print("\nMovies shape:", movies.shape)
print(movies.head())

# ── 2. Merge datasets ─────────────────────────────────────
df = pd.merge(ratings, movies, on='movieId')
print("\nMerged shape:", df.shape)

# ── 3. EDA — Long Tail Distribution ───────────────────────
movie_rating_counts = df.groupby('title')['rating'].count().sort_values(ascending=False)

plt.figure(figsize=(12, 4))
plt.plot(range(len(movie_rating_counts)), movie_rating_counts.values)
plt.title('Long Tail Distribution of Movie Popularity')
plt.xlabel('Movie Rank')
plt.ylabel('Number of Ratings')
plt.tight_layout()
plt.savefig('long_tail.png')
plt.show()

# ── 4. Rating Distribution ────────────────────────────────
plt.figure(figsize=(8, 4))
sns.countplot(x='rating', data=df)
plt.title('Rating Distribution')
plt.savefig('rating_dist.png')
plt.show()

# ── 5. Build User-Item Utility Matrix ─────────────────────
utility_matrix = df.pivot_table(
    index='userId',
    columns='title',
    values='rating'
)
print("\nUtility Matrix shape:", utility_matrix.shape)
print(utility_matrix.iloc[:5, :5])

# ── 6. Calculate Sparsity ─────────────────────────────────
total_cells    = utility_matrix.shape[0] * utility_matrix.shape[1]
filled_cells   = utility_matrix.notna().sum().sum()
sparsity       = 1 - (filled_cells / total_cells)
print(f"\nSparsity Ratio: {sparsity:.4f} ({sparsity*100:.2f}% empty)")

# Save for next phases
utility_matrix.to_csv('utility_matrix.csv')
df.to_csv('merged_data.csv', index=False)
print("\nPhase 1 Complete ✅")