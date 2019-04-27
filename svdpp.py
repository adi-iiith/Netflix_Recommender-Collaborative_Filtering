import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from datetime import datetime
import os
import itertools
import math

file_name = "../input/ml-100k.data"

def mapping(train) :
    uid_dict = {}
    iid_dict = {}
    current_u_index = 0
    current_i_index = 0

    row = []
    col = []
    data = []
    for urid,irid,r,timestamp in train :

    	try:
    		uid = uid_dict[urid]
    	except KeyError:
    		uid = current_u_index
    		uid_dict[urid] = current_u_index
    		current_u_index += 1
    	try:
    		iid = iid_dict[irid]
    	except KeyError:
    		iid = current_i_index
    		iid_dict[irid] = current_i_index
    		current_i_index += 1

    	row.append(uid)
    	col.append(iid)
    	data.append(r)

    train_sparse = csr_matrix((data, (row, col)))
    return train_sparse, uid_dict, iid_dict
    
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
		np.random.shuffle(raw_ratings)

	raw_len = len(raw_ratings)

	train_sparse,uid,iid = mapping(raw_ratings[:math.ceil(raw_len*0.8)])
	test = raw_ratings[math.ceil(raw_len*0.8):]

	return train_sparse,uid,iid,test


def all_ratings(matrix,axis=1):
    """
    row(u,i,r)
    or 
    col(u, i, r)
    """
    coo_matrix = matrix.tocoo()

    if axis == 1:
        return zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)
    else:
        return coo_matrix.row, coo_matrix.col, coo_matrix.data

def get_user(matrix, u):
    """
    (u, (is, rs)) 
    """

    ratings = matrix.getrow(u).tocoo()
    return ratings.col, ratings.data

def get_item(matrix, i):
    """
    (is, (us, rs))
    """

    ratings = matrix.getcol(i).tocoo()
    return ratings.row, ratings.data


def get_item_means(matrix):


    item_means = {}
    for i in np.unique(matrix.tocoo().col) :
        item_means[i] = np.mean(get_item(matrix,i)[1])
    return item_means

def get_user_means(matrix):


    users_mean = {}
    for u in np.unique(matrix.tocoo().row) :
        users_mean[u] = np.mean(get_user(matrix,u)[1])
    return users_mean


