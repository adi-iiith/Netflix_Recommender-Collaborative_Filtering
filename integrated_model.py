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
   
    item_means = {}
    for i in np.unique(matrix.tocoo().col) :
        item_means[i] = np.mean(get_item(matrix,i)[1])
    return item_means

def get_user_means(matrix):
  
    users_mean = {}
    for u in np.unique(matrix.tocoo().row) :
        users_mean[u] = np.mean(get_user(matrix,u)[1])
    return users_mean



def train(train_sparse,test, n_epochs = 30, n_factors = 20) :

    matrix = train_sparse.tocsc()
    user_num = matrix.shape[0]
    item_num = matrix.shape[1]

    #global mean
    global_mean = np.sum(matrix.data) / matrix.size

    #user bias
    bu = np.zeros(user_num, np.double)

    #item bias
    bi = np.zeros(item_num, np.double)

    #user factor
    p = np.zeros((user_num, n_factors), np.double) + .1

    #item factor
    q = np.zeros((item_num, n_factors), np.double) + .1

    #item preference facotor
    y = np.zeros((item_num, n_factors), np.double) + .1

    #weights for neihbourhood
    w = np.zeros((item_num,item_num))

    #implicit feedback
    c = np.zeros((item_num,item_num))



    n_lr = 0.001
    lr = 0.007
    reg = 0.001
    n_reg = 0.015

    reg7 = 0.005

    for current_epoch in range(n_epochs):
        start = datetime.now()
        print(" processing epoch {}".format(current_epoch))
        
        for u,i,r in all_ratings(matrix):
            
            Nu = get_user(matrix,u)[0]
            I_Nu = len(Nu)
            sqrt_N_u = np.sqrt(I_Nu)

            
            y_u = np.sum(y[Nu], axis=0)

            u_impl_prf = y_u / sqrt_N_u

            c_ij = np.sum(c[i,Nu] , axis = 0)

            w_ij = np.dot((get_user(matrix,u)[1] - global_mean - bu[u] - bi[Nu]) ,w[i][Nu])


            c_w =  (c_ij + w_ij )/sqrt_N_u

           
            rp = global_mean + bu[u] + bi[i] + np.dot(q[i], p[u] + u_impl_prf) + c_w

            
            e_ui = r - rp


            #sgd


            bu[u] += lr * (e_ui - reg7 * bu[u])
            bi[i] += lr * (e_ui - reg7 * bi[i])
            p[u] += lr * (e_ui * q[i] - reg * p[u])
            q[i] += lr * (e_ui * (p[u] + u_impl_prf) - reg * q[i])
            for j in Nu:
                y[j] += lr * (e_ui * q[j] / sqrt_N_u - reg * y[j])
            for j in Nu :
                w[i,j] += n_lr * (e_ui/ sqrt_N_u * (r - global_mean - bu[u] - bi[j]) - n_reg * w[i,j])
            for j in Nu :
                c[i,j] += n_lr * ((e_ui / sqrt_N_u) - n_reg * c[i,j])




        n_lr *= 0.9
        lr *= 0.9
        print("Time For Epoch :: "+str(datetime.now()-start))
     
    return bu,bi,y,c,w,q,p,global_mean

train_dataset, uid_dict, iid_dict, test_dataset = Read_Data(file_name,True)
bu,bi,y,c,w,q,p,global_mean = train(train_dataset,test_dataset,30)
np.savez("../integrated_model",bu,bi,y,c,w,q,p,global_mean)


