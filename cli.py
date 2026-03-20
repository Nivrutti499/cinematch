# cli.py - Command Line Interface for CineMatch
import pandas as pd
import numpy as np
import pickle

print("=" * 50)
print("   🎬 CineMatch Recommendation System")
print("=" * 50)

# Load all files
print("\nLoading models...")
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
                pred = svd_model.predict(user_id, mid)
                scores[title] = pred.est

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
                        w = sim.values
                        r = user_rated[sim.index].values
                        scores[title] = np.dot(w, r) / (np.sum(np.abs(w)) + 1e-9)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

def cold_start(n=5):
    popular = ratings.groupby('movieId')['rating'].count().nlargest(n).index
    return movies[movies['movieId'].isin(popular)]['title'].tolist()

# ── Main Menu Loop ────────────────────────────────────────
while True:
    print("\n" + "=" * 50)
    print("  MENU")
    print("  1. Get movie recommendations for a user")
    print("  2. See popular movies (for new users)")
    print("  3. Compare all 3 algorithms")
    print("  4. Exit")
    print("=" * 50)

    choice = input("Enter your choice (1/2/3/4): ").strip()

    if choice == '1':
        try:
            user_id = int(input("Enter User ID (1-610): ").strip())
            algo    = input("Algorithm? (svd / user / item) [default: svd]: ").strip() or 'svd'
            n       = int(input("How many recommendations? [default: 5]: ").strip() or 5)

            print(f"\n🎬 Top {n} Recommendations for User #{user_id} using {algo.upper()}:")
            print("-" * 50)
            results = recommend_movies(user_id, n=n, model=algo)
            if results:
                for i, (title, score) in enumerate(results, 1):
                    print(f"  {i}. {title}")
                    print(f"     ⭐ Predicted Rating: {score:.2f}/5.00")
            else:
                print("  No recommendations found for this user/algorithm.")

            # Show what user already rated
            if user_id in utility_matrix.index:
                print(f"\n📽️  Movies User #{user_id} already rated:")
                rated = utility_matrix.loc[user_id].dropna().sort_values(ascending=False)
                for title, rating in rated.head(5).items():
                    print(f"  • {title}: {rating}/5")
        except ValueError:
            print("❌ Please enter a valid number!")

    elif choice == '2':
        print("\n🔥 Most Popular Movies (Cold Start):")
        print("-" * 50)
        for i, title in enumerate(cold_start(10), 1):
            print(f"  {i}. {title}")

    elif choice == '3':
        try:
            user_id = int(input("Enter User ID (1-610): ").strip())
            print(f"\n📊 Comparing all algorithms for User #{user_id}:")
            for algo in ['svd', 'user', 'item']:
                print(f"\n  🔹 {algo.upper()} Top 3:")
                results = recommend_movies(user_id, n=3, model=algo)
                if results:
                    for i, (title, score) in enumerate(results, 1):
                        print(f"    {i}. {title} ({score:.2f}⭐)")
                else:
                    print("    No results found.")
        except ValueError:
            print("❌ Please enter a valid number!")

    elif choice == '4':
        print("\n👋 Goodbye! Thanks for using CineMatch!")
        break
    else:
        print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")