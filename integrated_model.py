import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from datetime import datetime
import os
import itertools
import math

file_name = "../input/ml-100k.data"

def parse_line(line):
    line = line.split("\t")
    uid, iid, r, timestamp = (line[i].strip() for i in range(4))
    return uid, iid, float(r), timestamp

def Read_Data(file_name,shuffle=True) :

	# data = pd.read_csv(file_name,sep = "\t", names = ["uid","iid","r","timst"])
	# data.sort_values(by = "timst", inplace = True)

	# train = data.head(n = int(0.8 * data.shape[0]))
	# test = data.drop(train.index)
	with open(os.path.expanduser(file_name)) as f:
		raw_ratings = [parse_line(line) for line in itertools.islice(f, 0, None)]
	if shuffle:
		# train_test shuffle in reproducable manner
		np.random.seed(73)
		np.random.shuffle(raw_ratings)

	raw_len = len(raw_ratings)

	train_sparse,uid,iid = mapping(raw_ratings[:math.ceil(raw_len*0.8)])
	test = raw_ratings[math.ceil(raw_len*0.8):]
	return train_sparse,uid,iid,test


def all_ratings(matrix,axis=1):

    coo_matrix = matrix.tocoo()

    if axis == 1:
        return zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)
    else:
        return coo_matrix.row, coo_matrix.col, coo_matrix.data

def get_user(matrix, u):

    ratings = matrix.getrow(u).tocoo()
    return ratings.col, ratings.data

def get_item(matrix, i):
    ratings = matrix.getcol(i).tocoo()
    return ratings.row, ratings.data

def get_item_means(matrix):
    """
    """
    item_means = {}
    for i in np.unique(matrix.tocoo().col) :
        item_means[i] = np.mean(get_item(matrix,i)[1])
    return item_means

def get_user_means(matrix):
    """
    """
    users_mean = {}
    for u in np.unique(matrix.tocoo().row) :
        users_mean[u] = np.mean(get_user(matrix,u)[1])
    return users_mean


train_dataset, uid_dict, iid_dict, test_dataset = Read_Data(file_name,True)
