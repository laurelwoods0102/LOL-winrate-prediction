import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from collections import defaultdict

import tensorflow as tf
from tensorflow import keras

def df_to_dataset(dataframe, batch_size=32):
    dataframe = dataframe.copy()
    response = dataframe.pop('y')    
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), response.values))

    #if shuffle:
        #dataset = dataset.shuffle(buffer_size=len(dataframe))
    dataset = dataset.batch(batch_size)

    return dataset

def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)

    labels = tf.dtypes.cast(labels, tf.float32) # reshape Labels
    labels = tf.reshape(labels, [32, 1])
    labels = tf.transpose(labels)

    return features, labels

def input_to_dataset(dataframe):
    response = dataframe.pop('y')
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), response.values))

    def input_pack_features_vector(features, labels):
        features = tf.stack(list(features.values()), axis=1)
    dataset = dataset.map(input_pack_features_vector)

    return dataset

class LogisticLayer(keras.layers.Layer):
    def __init__(self, input_dim=146, num_outputs=1):
        super(LogisticLayer, self).__init__()
        w_init = tf.initializers.GlorotUniform()     # NOTICE : weight matrix itself contains bias
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, num_outputs), dtype='float32'),
            trainable=True)

    def call(self, features):
        return tf.matmul(features, self.w)

class LogisticModel(tf.keras.Model):
    def __init__(self):
        super(LogisticModel, self).__init__()
        self.logistic_layer = LogisticLayer()

    @tf.function
    def call(self, features, training=False):
        x = self.logistic_layer(features)
        return x


optimizer = tf.optimizers.Adam(learning_rate=0.01)

def hypothesis(logit):
    return tf.divide(1.0, 1.0 + tf.exp(-1.0*logit))

def cost(logits, labels):
    hypo = hypothesis(logits)
    return -tf.reduce_mean(tf.math.log(hypo)*labels + tf.math.log(1.0-hypo)*(1.0-labels))

#def grad(logits, labels):
def grad(model, logits, labels):
    with tf.GradientTape() as tape:
        #cost_value = cost(model(features), labels)
        cost_value = cost(logits, labels)
    return cost_value, tape.gradient(cost_value, model.trainable_variables)

def train(model, features, labels):
    with tf.GradientTape() as t:
        current_cost = cost(model(features), labels)
    dW = t.gradient(current_cost, [model.logistic_layer.w])
    #print("current W : ", model.logistic_layer.w)
    optimizer.apply_gradients(zip(dW, [model.logistic_layer.w]))
    #print("trained W : ", model.logistic_layer.w)    
    
    return current_cost

def batch_accuracy(hypos, labels, batch_size=32):
    accuracy = 0
    predictor = zip(hypos, labels)
    for hypo, label in predictor:
        prediction = 0
        if hypo >= 0/5: prediction = 1

        if prediction == label: 
            accuracy += 1
        #print(hypo, " => ", prediction, " : ", label)
    return accuracy/batch_size


def KFoldValidation(df):
    batch_size = 32

    train_len, test_size = divmod(len(df), batch_size)
    if test_size == 0:
        train_len -= 1
        test_size = batch_size

    train_df = df.loc[:train_len*batch_size-1]
    test_df = df.loc[train_len*batch_size-1:]

    models = list()
    train_cost = list()
    val_accuracy = list()

    kf = KFold(n_splits=train_len)
    for train_index, val_index in kf.split(train_df):
        model = LogisticModel()

        train_ds = df_to_dataset(train_df.iloc[train_index])
        train_dataset = train_ds.map(pack_features_vector)
        for features, labels in train_dataset:
            current_cost = train(model, features, labels)
            train_cost.append(current_cost)

        val_ds = df_to_dataset(train_df.iloc[val_index])
        val_dataset = val_ds.map(pack_features_vector)
        for features, labels in val_dataset:
            logits = model(features)
            hypos = hypothesis(logits)
            
            accuracy = batch_accuracy(tf.reshape(hypos, [1, 32]).numpy()[0], labels.numpy()[0])
            val_accuracy.append(accuracy)
        
        models.append(model)
    
    print("Train Cost", train_cost)
    print("Val Accuracy", val_accuracy)

if __name__ == "__main__":
    df = pd.read_csv("./dataset/dataset_hide-on-bush_enemy.csv")
    df = df.astype(float)
    df = df.sample(frac=1).reset_index(drop=True)   # shuffle

    #KFoldValidation(df)

    batch_size = 32

    train_len, test_size = divmod(len(df), batch_size)
    if test_size == 0:
        train_len -= 1
        test_size = batch_size

    train_df = df.loc[:train_len*batch_size-1]
    test_df = df.loc[train_len*batch_size-1:]

    train_ds = df_to_dataset(train_df)
    train_dataset = train_ds.map(pack_features_vector)

    model = LogisticModel()
    
    
    costs = list()
    for features, labels in train_dataset:
        current_cost = train(model, features, labels)
        costs.append(current_cost.numpy())
    
    call = model.call.get_concrete_function(tf.TensorSpec(None, tf.float32))
    tf.saved_model.save(model, "./trained_model/", signatures=call)
    print(costs)