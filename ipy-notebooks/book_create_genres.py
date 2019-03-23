import os 
import csv
from collections import defaultdict
import itertools
import ast
import numpy as np 
import pandas as pd
import pickle
import sys


def create_final_movie_dict(csv_file, ratings_threshold, usernum_threshold):
    # using usecols in read_csv to exclude the last column of timestamp
    df = pd.read_csv(csv_file, delimiter=',', usecols=[0,1,2])    
    df = df[df['rating'] >= ratings_threshold]  # rating thresholding
    
    book_user_dict = defaultdict(list)
    for index, row in df.iterrows():    
        bookId = int(row['book_id'])
        usrId = int(row['user_id'])
        # rating = row['rating']       
        book_user_dict[bookId].append(usrId)

    # store the final res here, after the usernum thresholding
    final_dict = {}
    # num_users = 0
    for key,val in book_user_dict.items():
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
        book1, book2 = pair
        count_matrix[pair] = findnum_common_elements(final_dict[book1], final_dict[book2])
    
    # adding the marginal counts
    for k in final_dict.keys():
        pair = (k,k)
        count_matrix[pair] = len(final_dict[k])
    
    return count_matrix


def get_genres(genres_json, id_to_genre):
    '''
    id_to_genre: its like a global dict to store the mappings 
                from the genre-id to genre-name
    '''
    if not genres_json:
        return None
    
    tmp = ast.literal_eval(genres_json)
    ids_list = []
    gen_list = []           
    
    for d in tmp:
        id_num = d['id']
        genre_name = d['name']
        genre_name = genre_name.replace(" ", "-") # change all whitespaces to hyphen
        ids_list.append(id_num)
        gen_list.append(genre_name)
        
        if id_num not in id_to_genre:
            id_to_genre[id_num] = genre_name
    
    return ids_list, gen_list, id_to_genre

def create_final_genres_dict(relevent_genres, tags_csv, books_csv, book_tags_csv, tag_count_threshold):
    
    
    tmp_genres = []
    with open(relevent_genres, 'r') as f:    
        for l in f.readlines():
            t = l.strip()
            print(t)
            tmp_genres.append(t)
    #         print l.strip()

    tmp_genres[0] = '10th-century'
    good_genres = [ val for i,val in enumerate(tmp_genres) if i%2 == 0]

    df_tags = pd.read_csv(tags_csv, delimiter=',')
    df_tags.dataframeName = 'tags.csv'
    
    tag_id_list = df_tags['tag_id'].tolist()
    tag_name_list = df_tags['tag_name'].tolist()

    tag_id_to_name = {}
    tag_name_to_id = {}
    N = len(tag_id_list)

    for i in range(N):
        tagid = tag_id_list[i]
        tagnm = tag_name_list[i]
        
        if tagnm in good_genres:
            tag_id_to_name[tagid] = tagnm
            tag_name_to_id[tagnm] = tagid

    good_tag_ids = list(tag_id_to_name.keys())
    good_tag_names = list(tag_name_to_id.keys())    

    df_books = pd.read_csv(books_csv, delimiter=',')
    df_books.dataframeName = 'books.csv'
    df_books= df_books[['book_id', 'goodreads_book_id']]    

    #dict with key: goodreads_book_id, val: book_id 
    book_id_dict = defaultdict(lambda:-1)
    for index, row in df_books.iterrows():    
        bookId = int(row['book_id'])
        GoodReadId = int(row['goodreads_book_id'])
        book_id_dict[GoodReadId]=bookId

    #load book_tags file
    df_book_tags = pd.read_csv(book_tags_csv, delimiter=',')
    df_book_tags.dataframeName = 'book_tags.csv'    
    # tag_count_threshold = 500
    df_book_tags = df_book_tags[df_book_tags['count'] >= tag_count_threshold]

    for index, row in df_book_tags.iterrows():
        bookId = book_id_dict[int(row['goodreads_book_id'])]
        df_book_tags.set_value(index,'goodreads_book_id', bookId) 
    print(df_book_tags.head(5))

    df_book_tags.rename(columns={'goodreads_book_id': 'book_id'}, inplace=True)
    print(df_book_tags.head(5))

    book_tag_dict = defaultdict(list)
    for index, row in df_book_tags.iterrows():    
        bookId = int(row['book_id'])
        tagId = int(row['tag_id'])
        if tagId in tag_id_to_name: # Only conisder the tag if its in our good-tag list.
            book_tag_dict[bookId].append(tagId)
    print(len(book_tag_dict))

    num_books = len(book_tag_dict.keys())

    




    




    
    df_meta = pd.read_csv(csv_file, delimiter=',')    
    df_meta = df_meta[['genres', 'id']]
    
    genid_dict = {}
    genname_dict = {}
    genid_counts = defaultdict(int) # counting the genre-genre pairs
    id_to_genre = {}

    for index, row in df_meta.iterrows():    
        
        try:
            movId = int(row['id'])
            gen_json = row['genres']

            if movId in movIdsList:  # Only consider the thresholded movies
                ids, genres, id_to_genre = get_genres(gen_json, id_to_genre)    
                genid_dict[movId] = ids
                genname_dict[movId] = genres

                for i in ids:
                    singleton_pair = (i, i)
                    genid_counts[singleton_pair] += 1

                for pair in itertools.combinations(ids, r=2):
                    id1, id2 = pair
                    genid_counts[pair] += 1
                
        except Exception as e:
            print(index, row['id'])
            repr(e)
    
    return genid_dict, genname_dict, genid_counts, id_to_genre





def marginal_prob(obj_id, count_matrix, num_users):
    '''function to get the marginal prob:
        P(movie_id1)       
    '''
    margn_count = count_matrix[(obj_id, obj_id)]
    return margn_count * 1.0/num_users

def joint_prob(obj_id1, obj_id2, count_matrix, num_users):
    '''function to get the joint prob:
        P(movie_id1, movie_id2)       
    '''
    if (obj_id1, obj_id2) not in count_matrix.keys():
        key = (obj_id2, obj_id1)
    else:
        key = (obj_id1, obj_id2)
    
    joint_count = count_matrix[key]    
    return joint_count * 1.0/num_users

def conditional_prob(obj_id1, obj_id2, count_matrix):
    '''function to get the conditional prob:
        P(movie_id1 | movie_id2)       
    '''
    if (obj_id1, obj_id2) not in count_matrix.keys():
        key = (obj_id2, obj_id1)
    else:
        key = (obj_id1, obj_id2)
    
    joint_count = count_matrix[key]
    margn_count = count_matrix[obj_id2, obj_id2]
    
    return joint_count * 1.0/margn_count


def create_vocab_marginal_files(count_matrix, num_users, datadir):
    # Create the marginals file and a separate vocab file (for all the possible movieIds)
    # Consider only the diagonal entries in count_matrix
    # Also, the 2 lists work as follows: 
    # marginals[i] == marginal_prob(movieid_vocab[i], count_matrix, num_users)
    # i.e the indices of both lists match on movieid and its marginal probability

    marginals = []
    bookid_vocab = []
    for key_pair in count_matrix.keys():
        k1, k2 = key_pair
        if k1 == k2:        
            bookid_vocab.append(k1)
            marginals.append(marginal_prob(k1, count_matrix, num_users))

    # Write out the lists to text files
    fname_marginals = datadir + "book_marginal_prob.txt"
    with open(fname_marginals, "w") as f:
        for prob in marginals:
            f.write("%s\n" % prob)

    fname_vocab = datadir + "vocabulary.txt"
    with open(fname_vocab, "w") as f:
        for movid in bookid_vocab:
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
    
    fname_master = datadir + "master_book_data.txt"
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
    fname = datadir + "book_train.txt"
    with open(fname, "w") as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerows(tup_list[:trn_split])
    
    # write out the dev file
    # tst_data = tup_list[dev_split:]    
    fname = datadir + "book_dev.txt"
    with open(fname, "w") as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerows(tup_list[trn_split : dev_split])
        
    # write out the test file
    # tst_data = tup_list[dev_split:]
    fname = datadir + "book_test.txt"
    with open(fname, "w") as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerows(tup_list[dev_split:])
    
    # write out the trn-tst file -- its the merger of train+dev for evaluating training
    # trn_tst_data = tup_list[:dev_split]
    fname = datadir + "book_train_test.txt"
    with open(fname, "w") as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerows(tup_list[:dev_split])

    return


def create_trneval_file(train_file, train_eval_file, percentage):

    with open(train_file, "r") as f:
        lines = f.readlines()

    count = len(lines)*percentage//100
    t = 2*np.random.choice(len(lines)//2, count//2, replace=False)

    print("Selecting %d%% of train data as eval content (%d rows)."%(percentage, count))
    with open(train_eval_file, "w") as f:
        for i in t:
            f.write(lines[i])
            f.write(lines[i+1])
    print("Finished writing to file:", train_eval_file)
    return


def main():
    # t_rating = 4
    # t_users = 100
    t_rating = float(sys.argv[1])
    t_users = float(sys.argv[2])    
    splits = [0.8, 0.1, 0.1]    # the trn, dev and tst splits of data

    rootdir = '/home/ninad/Desktop/Link-to-sem4/dsis/'    
    # rawdata_file = rootdir + 'datasets/goodbooks-10k-master/samples/ratings.csv'
    rawdata_file = rootdir + 'datasets/goodbooks-10k-master/ratings.csv'
    datadir = rootdir + 'prob-emb/box-code/data/book_data/book_data_' + str(t_rating) + '_' + str(t_users) + '/'
    
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
