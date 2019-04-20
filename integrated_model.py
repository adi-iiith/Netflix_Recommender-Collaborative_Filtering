import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from datetime import datetime
import os
import itertools
import math

file_name = "../input/ml-100k.data"


def Read_Data(file_name,shuffle=True) :

	data = pd.read_csv(file_name,sep = "\t", names = ["uid","iid","r","timst"])
	data.sort_values(by = "timst", inplace = True)

	train = data.head(n = int(0.8 * da