import numpy as np
from scipy.sparse import csr_matrix
from util.matrix import Matrix
import util.tools as tl
import pandas as pd
from datetime import datetime
import os
import itertools
import math

file_name = "data/ml-100k/u.data"

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
	test = raw_ratings[math.ceil(raw_len*0.8):]
	return train_sparse,uid,iid,test

train_dataset, uid_dict, iid_dict, test_dataset = Read_Data(file_name,True)

npzfile = np.load("../integrated_model")

bu 			= npzfile['arr_0']
bi 			= npzfile['arr_1']
y  			= npzfile['arr_2']
c  			= npzfile['arr_3']
w  			= npzfile['arr_4']
q  			= npzfile['arr_5']
p           = npzfile['arr_6']
global_mean = npzfile['arr_7']

