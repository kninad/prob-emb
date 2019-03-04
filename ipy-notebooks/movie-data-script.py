import os 
import numpy as np 
import pandas as pd
from collections import defaultdict
import itertools
import ast

DATADIR = '../../data/the-movies-dataset/'

def get_df_from_csv(DATADIR, filename):
    # small_ratings_file = datadir + 'ratings_small.csv'
    # big_ratings_file = datadir + 'ratings.csv'
    csv_file = DATADIR + filename
    # using usecols in read_csv to exclude the last column of timestamp
    df = pd.read_csv(csv_file, delimiter=',', usecols=[0,1,2])
    return df

def create_final_dict(df, ratings_threshold, usernum_threshold):
    movie_user_dict = defaultdict(list)
    df = df[df['rating'] >= ratings_threshold]  # rating thresholding
    for index, row in df.iterrows():    
        movId = int(row['movieId'])
        usrId = int(row['userId'])
        # rating = row['rating']       
        movie_user_dict[movId].append(usrId)

    # store the final res here, after the usernum thresholding
    final_dict = {}
    # num_users = 0
    for key,val in movie_user_dict.items():
        if len(val) >= usernum_threshold:
            final_dict[key] = val
    
    return final_dict

def calc_total_users(final_dict):
    total_users = set()
    for mov in final_dict.keys():
        row = set(final_dict[mov])
        total_users = total_users.union(row)
    num_users = len(total_users)
    return num_users

def create_count_matrix(final_dict):
    # helper function
    def findnum_common_elements(list1, list2):
        return len(set(list1).intersection(list2))

    count_matrix = {}
    for pair in itertools.combinations(final_dict.keys(), r=2):
        mov1, mov2 = pair
        count_matrix[pair] = findnum_common_elements(final_dict[mov1], final_dict[mov2])
    
    # adding the marginal counts
    for k in final_dict.keys():
        pair = (k,k)
        count_matrix[pair] = len(final_dict[k])


def marginal_prob(movie_id, count_matrix, num_users):
    '''function to get the marginal prob:
        P(movie_id1)       
    '''
    margn_count = count_matrix[(movie_id, movie_id)]
    return margn_count * 1.0/num_users

def joint_prob(movie_id1, movie_id2, count_matrix, num_users):
    '''function to get the joint prob:
        P(movie_id1, movie_id2)       
    '''
    if (movie_id1, movie_id2) not in count_matrix.keys():
        key = (movie_id2, movie_id1)
    else:
        key = (movie_id1, movie_id2)
    
    joint_count = count_matrix[key]    
    return joint_count * 1.0/num_users


def conditional_prob(movie_id1, movie_id2, count_matrix):
    '''function to get the conditional prob:
        P(movie_id1 | movie_id2)       
    '''
    if (movie_id1, movie_id2) not in count_matrix.keys():
        key = (movie_id2, movie_id1)
    else:
        key = (movie_id1, movie_id2)
    
    joint_count = count_matrix[key]
    margn_count = count_matrix[movie_id2, movie_id2]
    
    return joint_count * 1.0/margn_count