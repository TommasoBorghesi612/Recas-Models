import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime
import tensorflow as tf 

sys.stdout = open('load_out.txt', 'w')
sys.stderr = open('load_err.txt', 'w') 

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

temp = np.array(pd.read_hdf('/lustrehome/tborghes/compact_data/data_1.h5', key='data', stop=1000000), dtype = np.float32)

print('loading 1 complete')

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

temp2 = np.array(pd.read_hdf('/lustrehome/tborghes/compact_data/data_1.h5', key='data', stop=1500000), dtype = np.float32)

print('loading 2 complete')

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

temp3 = np.array(pd.read_hdf('/lustrehome/tborghes/compact_data/data_1.h5', key='data', stop=2000000), dtype = np.float32)

print('loading 3 complete')

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

temp4 = np.array(pd.read_hdf('/lustrehome/tborghes/compact_data/data_1.h5', key='data', start = 3000000), dtype = np.float32)

print('loading 4 complete')

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

con_small = tf.concat([temp, temp2], axis = 0)

print('concat 1 complete')

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

con_small = tf.concat([temp, temp2], axis = 0)

print('concat 2 complete')

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

con_small = tf.concat([temp, temp2], axis = 0)

print('concat 3 complete')

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

con_large = tf.concat([temp, temp2, temp3, temp4], axis = 0)

print('concat 4 complete')

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

