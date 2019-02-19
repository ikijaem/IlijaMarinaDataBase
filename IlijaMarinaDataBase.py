import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


#variables containg data

def loadData():
    # Reading users file:
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('data/u.user', sep='|', names=u_cols, encoding='latin-1')

    # Reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('data/u.data', sep='\t', names=r_cols, encoding='latin-1')

    # Reading items file:
    i_cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action',
              'Adventure',
              'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv('data/u.item', sep='|', names=i_cols,
                        encoding='latin-1')
    return users, ratings, items




if __name__ == '__main__':
    users, ratings, items = loadData();
    print (users.shape)
