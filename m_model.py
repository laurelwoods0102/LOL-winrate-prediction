import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

df = pd.read_csv("./dataset/dataset_hide-on-bush_myPick_ex.csv")

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


#class LogisticLayer(keras.layers.Layer):
class LogisticRegression(object):
    def __init__(self, input_dim=146):    # 145 columns + 1 bias col
        #super(LogisticLayer, self).__init__()
        w_init = tf.random_normal_initializer()     # NOTICE : weight matrix itself contains bias
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim,), dtype='float32'),
            trainable=True)

    def __call__(self, input_features):
        return tf.matmul(input_features, self.w)



def cost(logit, input_labels):
    hypothesis = tf.divide(1.0, 1.0 + tf.exp(-1.0*logit))
    return -1.0*tf.reduce_mean(input_labels*tf.math.log(hypothesis) + (1-input_labels)*tf.math.log(1-hypothesis))

def train(model, input_features, input_labels, learning_rate=0.001):
    with tf.GradientTape() as t:
        current_loss = cost(model(input_features), input_labels)
    dW = t.gradient(current_loss, [model.W])
    model.W.assign_sub(learning_rate * dW)


model = LogisticRegression()
Ws = list()
epochs = range(2)
for epoch in epochs:
    for train_features, train_labels in train_ds:
        Ws.append(model.w.numpy())
        current_cost = cost(model(train_features), train_labels)

        train(model, train_features, train_labels)
        print('에포크 %2d: W=%1.2f 손실=%2.5f' %(epoch, Ws[-1], current_cost))


'''
class LogisticRegression(object):
    def __init__(self, units=32, input_dim=146):
        W_init = tf.random_normal_initializer()     # NOTICE : weight matrix itself contains bias
        self.W = tf.Variable(
            initial_value=W_init(shape=(input_dim, units), dtype='float32'),
            trainable=True)
    
    #@tf.function
    def forward(self, inputs):
        return tf.matmul(inputs, self.W)

    def __call__(self, inputs):
        return self.forward(inputs)
    
    def hypothesis(self, inputs):
        return tf.divide(1.0, 1.0 + tf.exp(-1.0*self.forward(inputs)))

    def cost(self, input_features, input_lables):
        hypothesis = self.hypothesis(input_features)
        return -1.0*tf.reduce_mean(input_lables*tf.math.log(hypothesis)+(1-input_lables)*tf.math.log(1-hypothesis))


class LogisticRegression(tf.keras.Model):
    def __init__(self, kernel_size=32, input_dim=146):
        W_init = tf.random_normal_initializer()     # NOTICE : weight matrix itself contains bias
        self.W = tf.Variable(
            initial_value=W_init(shape=(input_dim, units), dtype='float32'),
            trainable=True)
'''