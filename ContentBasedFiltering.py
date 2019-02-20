import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity

class ContentBasedFiltering:

    def __init__(self, movies):

        self.movies = movies

        genres = movies[['Action',
                        'Adventure',
                        'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                        'Western']].copy()

        self.matrix_similarity = cosine_similarity(genres, genres)

        self.indexes = pd.Series(movies.index, index=movies['movie_id']).drop_duplicates()

    def findRecommendations(self, movie):

        # get similarities with other movies of selected

        index_od_movie = self.indexes[movie]
        similarities = pd.Series(self.matrix_similarity[index_od_movie]).sort_values(ascending=False)

        # get top 10 results
        recommended_movies = []
        for i in list(similarities.index):
            if (index_od_movie == i):
                continue;
            recommended_movies.append(self.movies.at[i, 'movie_title'])
            if (len(recommended_movies) == 10):
                break;

        return pd.Series(recommended_movies)