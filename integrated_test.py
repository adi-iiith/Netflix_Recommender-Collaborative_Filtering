import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from datetime import datetime
import os
import itertools
import math

file_name = "../input/u.data"

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

	with open(os.path.expanduser(file_name)) as f:
		raw_ratings = [parse_line(line) for line in itertools.islice(f, 0, None)]
	if shuffle:
		np.random.shuffle(raw_ratings)

	raw_len = len(raw_ratings)

	train_sparse,uid,iid = mapping(raw_ratings[:math.ceil(raw_len*0.8)])
	test = raw_ratings[math.ceil(raw_len*0.8)-1:]
	return train_sparse,uid,iid,test

def get_user(matrix, u):
   
    ratings = matrix.getrow(u).tocoo()
    return ratings.col, ratings.data

def get_item(matrix, i):
    
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


def estimate(test, measures, train_sparse,bu,bi,y,c,w,q,p,global_mean):
	error = _estimate(test, measures, train_sparse,bu,bi,y,c,w,q,p,global_mean)
	error = np.sqrt(np.mean(np.power(error, 2)))
	return error

def convert_id(u,i,r):
	global uid_dict,iid_dict,user_raw,item_raw,i_rating
	user_raw.append(list(uid_dict.keys())[list(uid_dict.values()).index(u)])
	item_raw.append(list(iid_dict.keys())[list(iid_dict.values()).index(i)])
	i_rating.append(r)

def predict(matrix, u, i,bu,bi,y,c,w,q,p,global_mean):
	
	Nu = get_user(matrix,u)[0]

	I_Nu = len(Nu)
	sqrt_N_u = np.sqrt(I_Nu)

	y_u = np.sum(y[Nu], axis=0) / sqrt_N_u

	w_ij = np.dot((get_user(matrix,u)[1] - global_mean - bu[u] - bi[Nu]) ,w[i][Nu])
	c_ij = np.sum(c[i,Nu] , axis = 0)
	c_w =  (c_ij + w_ij )/sqrt_N_u


	est = global_mean + bu[u] + bi[i] + np.dot(q[i], p[u] + y_u) + c_w
	temp = min(5, est)
	temp = max(1, est)
	convert_id(u,i,temp)

	return est


def _estimate(test, measures, train_dataset,bu,bi,y,c,w,q,p,global_mean):
	global uid_dict,iid_dict

	users_mean = get_user_means(train_dataset)
	items_mean = get_item_means(train_dataset)

	raw_test_dataset = test
	global_mean = np.sum(train_dataset.data) / train_dataset.size


	all = len(raw_test_dataset)
	errors = []
	cur = 0
	alg_count = 0

	for raw_u, raw_i, r, _ in raw_test_dataset:
		cur += 1
		has_raw_u = raw_u in uid_dict
		has_raw_i = raw_i in iid_dict

		if not has_raw_u and not has_raw_i:
		    real, est = r, global_mean
		elif not has_raw_u:
		    i = iid_dict[raw_i]
		    real, est = r, items_mean[i]
		elif not has_raw_i:
		    u = uid_dict[raw_u]
		    real, est = r, users_mean[u]
		else:
		    u = uid_dict[raw_u]
		    i = iid_dict[raw_i]
		    real, est = r, predict(train_dataset,u, i,bu,bi,y,c,w,q,p,global_mean)
		    alg_count += 1

		est = min(5, est)
		est = max(1, est)
		errors.append(real - est)

	return errors


user_raw = list()
item_raw = list()
i_rating = list()

train_dataset, uid_dict, iid_dict, test_dataset = Read_Data(file_name,True)

npzfile = np.load("../integrated_model.npz")

bu 			= npzfile['arr_0']
bi 			= npzfile['arr_1']
y  			= npzfile['arr_2']
c  			= npzfile['arr_3']
w  			= npzfile['arr_4']
q  			= npzfile['arr_5']
p           = npzfile['arr_6']
global_mean = npzfile['arr_7']


error = estimate(test_dataset, "rmse", train_dataset,bu,bi,y,c,w,q,p,global_mean)

np.savez("../predictions",np.array(user_raw),np.array(item_raw),np.array(i_rating))

print("Error : ",error)