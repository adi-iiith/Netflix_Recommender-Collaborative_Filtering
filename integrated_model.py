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
		np.random.shuffle(raw_ratings)

	raw_len = len(raw_ratings)

	train_sparse,uid,iid = mapping(raw_ratings[:math.ceil(raw_len*0.8)])
	test = raw_ratings[math.ceil(raw_len*0.8):]
	return train_sparse,uid,iid,test
