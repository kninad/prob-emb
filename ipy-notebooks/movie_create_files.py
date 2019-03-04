import os 
import csv
from collections import defaultdict
import itertools
import ast
import numpy as np 
import pandas as pd

# def get_df_from_csv(DATADIR, filename):
#     csv_file = DATADIR + filename
#     # using usecols in read_csv to exclude the last column of timestamp
#     df = pd.read_csv(csv_file, delimiter=',', usecols=[0,1,2])
#     return df

def create_final_dict(csv_file, ratings_threshold, usernum_threshold):
    # using usecols in read_csv to exclude the last column of timestamp
    df = pd.read_csv(csv_file, delimiter=',', usecols=[0,1,2])    
    df = df[df['rating'] >= ratings_threshold]  # rating thresholding
    
    movie_user_dict = defaultdict(list)
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

def get_total_users(final_dict):
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
    
    return count_matrix

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


def create_vocab_marginal_files(count_matrix, num_users, datadir):
    # Create the marginals file and a separate vocab file (for all the possible movieIds)
    # Consider only the diagonal entries in count_matrix
    # Also, the 2 lists work as follows: 
    # marginals[i] == marginal_prob(movieid_vocab[i], count_matrix, num_users)
    # i.e the indices of both lists match on movieid and its marginal probability

    marginals = []
    movieid_vocab = []
    for key_pair in count_matrix.keys():
        k1, k2 = key_pair
        if k1 == k2:        
            movieid_vocab.append(k1)
            marginals.append(marginal_prob(k1, count_matrix, num_users))

    # Write out the lists to text files
    fname_marginals = datadir + "marginals.txt"
    with open(fname_marginals, "w") as f:
        for prob in marginals:
            f.write("%s\n" % prob)

    fname_vocab = datadir + "movie_vocab.txt"
    with open(fname_vocab, "w") as f:
        for movid in movieid_vocab:
            f.write("%s\n" % movid)

    return 


def create_trndevtst_files(count_matrix, splits, datadir):
    REL = "IsA" # Instead of 'IsA' relation, we use 'IsWith'. Nothing substantially different.
    # splits = [0.8, 0.1, 0.1]    # the trn, dev and tst splits of data

    tup_list = []
    for key_pair in count_matrix.keys():
        k1, k2 = key_pair
        if k1 != k2:
            prob_k2k1 = conditional_prob(k2, k1, count_matrix)
            tmp_tup1 = (REL, k1, k2, prob_k2k1)        
            tup_list.append(tmp_tup1)
            
            prob_k1k2 = conditional_prob(k1, k2, count_matrix)
            tmp_tup2 = (REL, k2, k1, prob_k1k2)
            tup_list.append(tmp_tup2)
    
    fname_master = datadir + "master_movie_data.txt"
    with open(fname_master, "w") as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerows(tup_list)

    total = len(tup_list)    
    trn_num = int(total * splits[0])
    dev_num = int(total * splits[1])
    tst_num = int(total * splits[2])

    if trn_num % 2 != 0:
        trn_num += 1
    trn_split = trn_num

    if dev_num % 2 != 0:
        dev_num += 1
    dev_split = trn_split + dev_num

    # REDUNDANT
    if tst_num % 2 != 0:
        tst_num += 1
    tst_split = dev_split + tst_num

    # trn_data = tup_list[:trn_split]
    # dev_data = tup_list[trn_split : dev_split]
    # trn_tst_data = tup_list[:dev_split]
    # tst_data = tup_list[dev_split:]


    # write out the training file
    # trn_data = tup_list[:trn_split]    
    fname = datadir + "movie_train.txt"
    with open(fname, "w") as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerows(tup_list[:trn_split])
    
    # write out the dev file
    # tst_data = tup_list[dev_split:]    
    fname = datadir + "movie_dev.txt"
    with open(fname, "w") as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerows(tup_list[trn_split : dev_split])
        
    # write out the test file
    # tst_data = tup_list[dev_split:]
    fname = datadir + "movie_test.txt"
    with open(fname, "w") as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerows(tup_list[dev_split:])
    
    # write out the trn-tst file -- its the merger of train+dev for evaluating training
    # trn_tst_data = tup_list[:dev_split]
    fname = datadir + "movie_train_test.txt"
    with open(fname, "w") as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerows(tup_list[:dev_split])

    return


def main():
    t_rating = 4
    t_users = 100
    splits = [0.8, 0.1, 0.1]    # the trn, dev and tst splits of data
    datadir = 'clean_data_movie_' + str(t_rating) + '_' + str(t_users) + '/'
    rawdata_file = '../../data/the-movies-dataset/ratings_small.csv'
    
    final_dict = create_final_dict(rawdata_file, t_rating, t_users)
    num_users = get_total_users(final_dict)
    cmatrix = create_count_matrix(final_dict)
    del(final_dict)
    
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    
    create_vocab_marginal_files(cmatrix, num_users, datadir)
    create_trndevtst_files(cmatrix, splits, datadir)


if __name__ == "__main__":
    main()