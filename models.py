import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

df = pd.read_csv("./dataset/dataset_hide-on-bush_myPick_ex.csv")
df = df.astype(float)

train_df, test_df = train_test_split(df, test_size=0.1)
train_df, val_df = train_test_split(train_df, test_size=0.1)

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    #dataframe = dataframe.copy()
    response = dataframe.pop('result')    
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), response.values))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    dataset = dataset.batch(batch_size)

    return dataset

train_ds = df_to_dataset(train_df, shuffle=True)
val_ds = df_to_dataset(val_df, shuffle=False)
test_ds = df_to_dataset(test_df, shuffle=False)

def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

train_dataset = train_ds.map(pack_features_vector)
#for features, labels in train_dataset.take(1):
#    print(features)


class LogisticLayer(keras.layers.Layer):
    def __init__(self, input_dim=146, num_outputs=1):
        super(LogisticLayer, self).__init__()
        w_init = tf.initializers.GlorotUniform()     # NOTICE : weight matrix itself contains bias
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, num_outputs), dtype='float32'),
            trainable=True)
        #print(self.w)

    def call(self, input_features):
        return tf.matmul(input_features, self.w)
'''
layer = LogisticLayer()
for features, lables in train_dataset.take(1):
    print("forward", layer(features))
    print("weights", layer.trainable_variables)
'''

class LogisticModel(tf.keras.Model):
    def __init__(self):
        super(LogisticModel, self).__init__()
        self.w = LogisticLayer()
    
    def call(self, input_features, training=False):
        x = self.w(input_features)
        return x

model = LogisticModel()
for features, labels in train_dataset.take(1):
    print(model.trainable_variables)
