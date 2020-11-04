import numpy as np
import pandas as pd


split_1 = np.array(pd.read_hdf('data_1', key='data', stop = 3000000), 
                    dtype = np.float32)

split_1 = pd.DataFrame(split_1)
split_1.to_hdf('split_1.h5', key='data', mode='w')

###

split_2a = np.array(pd.read_hdf('data_1', key='data', start = 3000000), 
                    dtype = np.float32)

split_2b_end = 3000000 - split_2a.shape[0]

split_2b = np.array(pd.read_hdf('data_2', key='data', stop = split_2b_end), 
                    dtype = np.float32)

print (split_2b_end)

split_2 = np.concatenate([split_2a, split_2b])

split_2 = pd.DataFrame(split_2)
split_2.to_hdf('split_2.h5', key='data', mode='w')

###

split_3a = np.array(pd.read_hdf('data_2', key='data', start = split_2b_end), 
                    dtype = np.float32)

split_3b_end = 3000000 - split_3a.shape[0]

split_3b = np.array(pd.read_hdf('data_3', key='data', stop = split_3b_end), 
                    dtype = np.float32)

print (split_3b_end)

split_3 = np.concatenate([split_3a, split_3b])

split_3 = pd.DataFrame(split_3)
split_3.to_hdf('split_3.h5', key='data', mode='w')

###

split_4a = np.array(pd.read_hdf('data_3', key='data', start = split_3b_end), 
                    dtype = np.float32)

split_4b_end = 3000000 - split_4a.shape[0]

split_4b = np.array(pd.read_hdf('data_4', key='data', stop = split_4b_end), 
                    dtype = np.float32)

print (split_4b_end)

split_4 = np.concatenate([split_4a, split_4b])

split_4 = pd.DataFrame(split_4)
split_4.to_hdf('split_4.h5', key='data', mode='w')

###

split_5 = np.array(pd.read_hdf('data_4', key='data', start = split_4b_end, 
                                stop = split_4b_end + 3000000), 
                                dtype = np.float32)

split_5 = pd.DataFrame(split_5)
split_5.to_hdf('split_5.h5', key='data', mode='w')


################################################

label_split_1 = np.array(pd.read_hdf('labels_1', key='labels', stop = 3000000), 
                    dtype = np.float32)

label_split_1 = pd.DataFrame(label_split_1)
label_split_1.to_hdf('label_split_1.h5', key='labels', mode='w')

###

label_split_2a = np.array(pd.read_hdf('labels_1', key='labels', start = 3000000), 
                    dtype = np.float32)

label_split_2b_end = 3000000 - label_split_2a.shape[0]

label_split_2b = np.array(pd.read_hdf('labels_2', key='labels', stop = label_split_2b_end), 
                    dtype = np.float32)

print (label_split_2b_end)

label_split_2 = np.concatenate([label_split_2a, label_split_2b])

label_split_2 = pd.DataFrame(label_split_2)
label_split_2.to_hdf('label_split_2.h5', key='labels', mode='w')

###

label_split_3a = np.array(pd.read_hdf('labels_2', key='labels', start = label_split_2b_end), 
                    dtype = np.float32)

label_split_3b_end = 3000000 - label_split_3a.shape[0]

label_split_3b = np.array(pd.read_hdf('labels_3', key='labels', stop = label_split_3b_end), 
                    dtype = np.float32)

print (label_split_3b_end)

label_split_3 = np.concatenate([label_split_3a, label_split_3b])

label_split_3 = pd.DataFrame(label_split_3)
label_split_3.to_hdf('label_split_3.h5', key='labels', mode='w')

###

label_split4a = np.array(pd.read_hdf('labels_3', key='labels', start = label_split_3b_end), 
                    dtype = np.float32)

label_split4b_end = 3000000 - label_split4a.shape[0]

label_split4b = np.array(pd.read_hdf('labels_4', key='labels', stop = label_split4b_end), 
                    dtype = np.float32)

print (label_split4b_end)

label_split4 = np.concatenate([label_split4a, label_split4b])

label_split4 = pd.DataFrame(label_split4)
label_split4.to_hdf('label_split4.h5', key='labels', mode='w')

###

label_split5 = np.array(pd.read_hdf('labels_4', key='labels', start = label_split4b_end, 
                                stop = label_split4b_end + 3000000), 
                                dtype = np.float32)

label_split5 = pd.DataFrame(label_split5)
label_split5.to_hdf('label_split5.h5', key='labels', mode='w')