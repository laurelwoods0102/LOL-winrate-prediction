import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras

'''
def df_to_dataset(dataframe, batch=True):
    #dataframe = dataframe.copy()
    response = dataframe.pop('y')    
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), response.values))

    #if shuffle:
        #dataset = dataset.shuffle(buffer_size=len(dataframe))
    if batch:
        batch_size = 32
        dataset = dataset.batch(batch_size)

    return dataset

def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)

    return features, labels


df = pd.read_csv("./dataset/dataset_hide-on-bush_enemy.csv")
df = df.astype(float)
df = df.sample(frac=1).reset_index(drop=True)

batch_size = 32

train_len, test_size = divmod(len(df), batch_size)
if test_size == 0:
    train_len -= 1
    test_size = batch_size

train_df = df.loc[:train_len*batch_size-1]
test_df = df.loc[train_len*batch_size-1:]

kf = KFold(n_splits=train_len)
for train_index, val_index in kf.split(train_df):
    #print(train_df.iloc[test_index])
    val_ds = df_to_dataset(train_df.iloc[val_index])
    val_dataset = val_ds.map(pack_features_vector)
    for f, l in val_dataset:
        print(l)
    break
'''

df = pd.read_csv("./dataset/test.csv")
df = df.astype(float)
df = df.sample(frac=1).reset_index(drop=True)

response = df.pop('y')
print(response.values)

for d in df.values:
    print(tf.convert_to_tensor(d))