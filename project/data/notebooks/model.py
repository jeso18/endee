import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, movies_path, ratings_path):
        self.movies_path = movies_path
        self.ratings_path = ratings_path
        self.data = None
        self.user_movie_matrix = None
        self.similarity_matrix = None

    def load_data(self):
        movies = pd.read_csv(self.movies_path)
        ratings = pd.read_csv(self.ratings_path)
        self.data = pd.merge(ratings, movies, on="movieId")

    def preprocess(self):
        self.user_movie_matrix = self.data.pivot_table(
            index="userId",
            columns="title",
            values="rating"
        ).fillna(0)

    def compute_similarity(self):
        self.similarity_matrix = cosine_similarity(self.user_movie_matrix)

    def fit(self):
        self.load_data()
        self.preprocess()
        self.compute_similarity()

    def recommend_movies(self, user_id, n=5, min_rating=4):
        if self.similarity_matrix is None:
            raise Exception("Model not trained. Call fit() first.")

        user_index = user_id - 1
        similarity_scores = list(enumerate(self.similarity_matrix[user_index]))

        # Sort users by similarity
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        recommended_movies = set()

        for idx, score in similarity_scores[1:]:
            user_ratings = self.user_movie_matrix.iloc[idx]
            high_rated_movies = user_ratings[user_ratings >= min_rating].index.tolist()

            recommended_movies.update(high_rated_movies)

            if len(recommended_movies) >= n:
                break

        return list(recommended_movies)[:n]
