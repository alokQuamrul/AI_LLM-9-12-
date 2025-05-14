import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from collections import defaultdict


class MovieRecommender:
    def __init__(self):
        # Load movie data
        self.movies = pd.read_csv('movies.csv')  # Assuming columns: movieId, title, genres
        self.ratings = pd.read_csv('ratings.csv')  # Assuming columns: userId, movieId, rating
        
        # Configure the rating scale
        reader = Reader(rating_scale= (1, 5))
        
        # Load the dataset
        data = Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']], reader)
        
        # Split the data into train and test set
        trainset, testset = train_test_split(data, test_size=0.25)
        
        # Use user-based collaborative filtering with cosine similarity
        sim_options = {
            'name': 'cosine',
            'user_based': True
        }
        
        # Train the KNN algorithm
        self.algo = KNNBasic(sim_options=sim_options)
        self.algo.fit(trainset)
        
        # Build anti-testset for generating recommendations
        self.trainset = trainset
    
    def get_top_n_recommendations(self, user_id, n=10):
        # Get a list of all movie IDs
        all_movie_ids = self.movies['movieId'].unique()
        
        # Get a list of movies the user has already rated
        rated_movies = self.ratings[self.ratings['userId'] == user_id]['movieId'].unique()
        
        # Get movies not rated by the user
        movies_to_predict = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movies]
        
        # Predict ratings for all movies not rated by user
        testset = [[user_id, movie_id, 4.] for movie_id in movies_to_predict]  # 4 is a dummy rating
        predictions = self.algo.test(testset)
        
        # Get the top N highest rated predictions
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))
        
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]
        
        # Get movie titles for the recommendations
        recommendations = []
        for movie_id, estimated_rating in top_n[user_id]:
            movie_title = self.movies[self.movies['movieId'] == movie_id]['title'].values[0]
            recommendations.append((movie_title, estimated_rating))
        
        return recommendations
    
    def get_similar_users(self, user_id, n=5):
        # Get similar users
        user_inner_id = self.trainset.to_inner_uid(user_id)
        user_neighbors = self.algo.get_neighbors(user_inner_id, k=n)
        
        similar_users = [self.trainset.to_raw_uid(inner_id) for inner_id in user_neighbors]
        return similar_users
    
    def get_similar_movies(self, movie_id, n=5):
        # Get similar movies
        try:
            movie_inner_id = self.trainset.to_inner_iid(movie_id)
            movie_neighbors = self.algo.get_neighbors(movie_inner_id, k=n)
            
            similar_movies = [self.trainset.to_raw_iid(inner_id) for inner_id in movie_neighbors]
            
            # Get movie titles
            similar_movie_titles = []
            for m_id in similar_movies:
                title = self.movies[self.movies['movieId'] == m_id]['title'].values[0]
                similar_movie_titles.append(title)
            
            return similar_movie_titles
        except ValueError:  # movie not in trainset
            return []

# Example usage
if __name__ == "__main__":
    recommender = MovieRecommender()
    
    # Get recommendations for user with ID 1
    user_id = 1
    print(f"Top 10 movie recommendations for user {user_id}:")
    recommendations = recommender.get_top_n_recommendations(user_id)
    for i, (title, rating) in enumerate(recommendations, 1):
        print(f"{i}. {title} (predicted rating: {rating:.2f})")
    
    # Get similar users
    similar_users = recommender.get_similar_users(user_id)
    print(f"\nUsers similar to user {user_id}: {similar_users}")
    
    # Get similar movies to a specific movie
    movie_id = 1  # Toy Story (1995) typically
    similar_movies = recommender.get_similar_movies(movie_id)
    print(f"\nMovies similar to movie ID {movie_id}:")
    for i, title in enumerate(similar_movies, 1):
        print(f"{i}. {title}")