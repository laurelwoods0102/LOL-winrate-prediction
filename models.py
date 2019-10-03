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

def hypothesis(logit):
    return tf.divide(1.0, 1.0 + tf.exp(-1.0*logit))

def cost(logit_tensor, input_labels):
    hypo = hypothesis(logit)
    likelihood_y_0 = tf.matmul(input_labels, tf.math.log(hypo))
    return -1.0*tf.reduce_mean(input_labels*tf.math.log(hypo) + (1.0-input_labels)*tf.math.log(1.0-hypo))

model = LogisticModel()
#optimizer = tf.keras.optimizer.Adam(learning_rate=0.01)
for features, labels in train_dataset.take(1):
    logit = model(features)
    labels = tf.dtypes.cast(labels, tf.float32)
    labels = tf.reshape(labels, [32, 1])
    labels = tf.transpose(labels)

    #print("hypothesis", hypothesis(logit))
    print(cost(logit, labels))
    '''
    result = list()
    for hypo in hypothesis(logit):
        prediction = 0
        if hypo[0] >= 0.5:
            prediction = 1
        result.append(prediction)
    print(result)
    '''