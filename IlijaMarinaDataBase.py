import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from ContentBasedFiltering import ContentBasedFiltering
from CollaborativeFIltering import CollaborativeFiltering


#variables containg data

def loadData():
    # Reading users file:
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('data/u.user', sep='|', names=u_cols, encoding='latin-1')

    # Reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('data/u.data', sep='\t', names=r_cols, encoding='latin-1')

    # Reading items file:
    m_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action',
              'Adventure',
              'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    movies = pd.read_csv('data/u.item', sep='|', names=m_cols,
                        encoding='latin-1')
    return users, ratings, movies

def randomizeUser(users):
    return random.randint(1, users.user_id.unique().shape[0])

def randomizeMovieLikedByUser(ratings, user):
    sorted_ratings = ratings.sort_values(by=['user_id'])
    users_movies = sorted_ratings.loc[sorted_ratings['user_id'] == user]
    movs = list(users_movies.movie_id)
    movie_idx = random.randint(0, len(movs)-1)
    return movs[movie_idx]

if __name__ == '__main__':
    users, ratings, movies = loadData();

    cbf = ContentBasedFiltering(movies)
    cf = CollaborativeFiltering(ratings, movies)

    #naredne 4 linije, random biraju film za nekog random usera, i onda na osnovu tog filma ce mu preporuciti
    user_id = randomizeUser(users)
    print (user_id)
    movie_id = randomizeMovieLikedByUser(ratings, user_id)
    print(movie_id)

    #content based
    #print (cbf.findRecommendations(movie_id))

    #collaborative
    print (cf.findRecommendations(603, 'user'))

