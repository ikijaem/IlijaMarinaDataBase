import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFiltering:

    def __init__(self, ratings, movies):
        self.ratings = ratings
        self.movies=movies

        n_users = ratings.user_id.unique().shape[0]
        n_movies = ratings.movie_id.unique().shape[0]

        self.data_matrix = np.zeros((n_users, n_movies))
        for line in ratings.itertuples():
            self.data_matrix[line[1] - 1, line[2] - 1] = line[3]

        self.user_similarity = cosine_similarity(self.data_matrix, self.data_matrix)
        self.item_similarity = cosine_similarity(self.data_matrix.T, self.data_matrix.T)


    def getMoviesUserRated(self, user):
        self.ratings = self.ratings.sort_values(by='user_id')
        user_ratings = self.ratings.loc[self.ratings['user_id'] == user]

        return list(user_ratings['movie_id'])

    def getPredictionMatrix(self, type):
        if (type == 'user') :
            similarity = self.user_similarity
            predictions_matrix = similarity.dot(self.data_matrix)
        else:
            similarity = self.item_similarity
            predictions_matrix = self.data_matrix.dot(similarity)

        #dec because each user is "similar" to himself
        sum_similarities= list([np.abs(similarity).sum(axis=1)])
        for i in range(0, len(sum_similarities[0])):
            sum_similarities[0][i] = sum_similarities[0][i]-1

        for i in range(0, len(predictions_matrix)):
            for j in range(0, len(predictions_matrix[0])):
                predictions_matrix[i, j] = predictions_matrix[i, j] / sum_similarities[0][i]

        return predictions_matrix

    def findRecommendations(self, user, type):
        predictions_matrix = self.getPredictionMatrix(type)
        predictions = pd.Series(predictions_matrix[user]).sort_values(ascending=False)

        movies_user_rated = self.getMoviesUserRated(user)
        # get top 10 results
        recommended_movies = []
        for i in list(predictions.index):
            if(i+1 in movies_user_rated):
                continue;
            recommended_movies.append(self.movies.at[i, 'movie_title'])
            if (len(recommended_movies) == 10):
                break;

        return recommended_movies