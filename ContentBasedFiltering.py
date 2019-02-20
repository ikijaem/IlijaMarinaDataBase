import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedFiltering:

    def __init__(self, movies, ratings):

        self.movies = movies
        self.ratings = ratings

        self.genres = movies[['Action',
                        'Adventure',
                        'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                        'Western']].copy()

    def getMoviesUserLiked(self, user):
        self.ratings = self.ratings.sort_values(by='user_id')
        temp = self.ratings.loc[self.ratings['user_id'] == user]
        user_likes = temp.loc[temp['rating'] >= 3]

        return list(user_likes['movie_id'])

    def findRecommendations(self, user):

        # get similarities with other movies of selected
        movies_liked = self.getMoviesUserLiked(user)
        matrix_liked = self.genres.loc[movies_liked]

        matrix = cosine_similarity(matrix_liked, self.genres)

        similar_movies = []
        #np.random.shuffle(matrix)
        for i in range (0, len(matrix)):
            all = pd.Series(matrix[i])

            while all.idxmax() in movies_liked or all.idxmax() in similar_movies:
                all = all.drop(all.idxmax())

            similar_movies.append(all.idxmax())
        similar_movies.sort(reverse=True)

        # get top 10 results
        recommended_movies = []

        for i in range(len(similar_movies)):
            if (i in movies_liked):
                continue;
            recommended_movies.append(self.movies.at[similar_movies[i], 'movie_title'])
            if (len(recommended_movies) == 10):
                break;

        return pd.Series(recommended_movies)