import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras


def df_to_dataset(dataframe, batch_size=32):
    #dataframe = dataframe.copy()
    response = dataframe.pop('y')    
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), response.values))

    #if shuffle:
        #dataset = dataset.shuffle(buffer_size=len(dataframe))
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

train_ds = df_to_dataset(train_df)
test_ds = df_to_dataset(test_df)

train_dataset = train_ds.map(pack_features_vector)
test_dataset = test_ds.map(pack_features_vector)

for i in range(train_len):
    for features, labels in train_dataset.skip(i):
        print("Training")
        print(features)
        print(labels)
        break
    for features, labels in train_dataset.take(i):
        print("Validation")
        print(features)
        print(labels)
        break

for features, labels in test_dataset:
    print("Test")
    print(features)
    print(labels)
    break

'''
train_ds = df_to_dataset(train_df, shuffle=True)
train_dataset = train_ds.map(pack_features_vector)

val = train_dataset.skip(1)
for f, l in val:
    print(f)
    print(l)
'''