import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from collections import defaultdict

import tensorflow as tf
from tensorflow import keras

def df_to_dataset(dataframe, batch_size=32):
    response = dataframe.pop('y')    
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), response.values))
    dataset = dataset.batch(batch_size)

    return dataset

def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)

    labels = tf.dtypes.cast(labels, tf.float32) # reshape Labels
    labels = tf.reshape(labels, [32, 1])
    labels = tf.transpose(labels)

    return features, labels

class LogisticLayer(keras.layers.Layer):
    def __init__(self, input_shape, num_outputs=1):
        super(LogisticLayer, self).__init__()
        w_init = tf.initializers.GlorotUniform()     # NOTICE : weight matrix itself contains bias
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_shape, num_outputs), dtype='float32'),
            trainable=True)

    @tf.function
    def call(self, features):
        return tf.matmul(features, self.w)

class LogisticModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(LogisticModel, self).__init__()
        self.logistic_layer = LogisticLayer(input_shape)    #input shape = 146, 4

    @tf.function
    def call(self, features, training=False):
        x = self.logistic_layer(features)
        return x

optimizer = tf.optimizers.Adam(learning_rate=0.1)

def hypothesis(logit):
    return tf.divide(1.0, 1.0 + tf.exp(-1.0*logit))

def cost(logits, labels):
    hypo = hypothesis(logits)
    return -tf.reduce_mean(tf.math.log(hypo)*labels + tf.math.log(1.0-hypo)*(1.0-labels))

def train(model, features, labels):
    with tf.GradientTape() as t:
        current_cost = cost(model(features), labels)
    dW = t.gradient(current_cost, [model.logistic_layer.w])
    optimizer.apply_gradients(zip(dW, [model.logistic_layer.w]))
    
    return current_cost

def batch_accuracy(hypos, labels, batch_size=32):
    accuracy = 0
    predictor = zip(hypos, labels)
    for hypo, label in predictor:
        prediction = 0
        if hypo >= 0/5: prediction = 1

        if prediction == label: 
            accuracy += 1
    return accuracy/batch_size


if __name__ == "__main__":
    name = "hide on bush"
    name = name.replace(" ", "-")

    df_team = pd.read_csv("./dataset/secondary/dataset_2_{0}_team.csv".format(name))
    df_enemy = pd.read_csv("./dataset/secondary/dataset_2_{0}_enemy.csv".format(name))
    df_average = pd.read_csv("./dataset/secondary/dataset_{0}_average.csv".format(name))
    df_response = pd.read_csv("./dataset/secondary/dataset_2_{0}_response.csv".format(name))

    bias = np.array([[float(1) for i in range(1088)]], dtype='f8')
    bias = pd.DataFrame(bias.T, columns=["bias"])
    
    train_df = pd.concat([bias, df_average, df_team, df_enemy, df_response], axis=1)
    #train_ds = df_to_dataset(train_df)
    #train_dataset = train_ds.map(pack_features_vector)
    '''
    for f, l in train_dataset.take(1):
        print(f)
        print(l)
    '''
    models = list()
    train_cost = list()
    val_accuracy = list()

    kf = KFold(n_splits=34)
    for train_index, val_index in kf.split(train_df):
        model = LogisticModel(4)

        train_ds = df_to_dataset(train_df.iloc[train_index])
        train_dataset = train_ds.map(pack_features_vector)
        for features, labels in train_dataset:
            current_cost = train(model, features, labels)
            train_cost.append(current_cost.numpy())

        val_ds = df_to_dataset(train_df.iloc[val_index])
        val_dataset = val_ds.map(pack_features_vector)
        for features, labels in val_dataset:
            logits = model(features)
            hypos = hypothesis(logits)
            
            accuracy = batch_accuracy(tf.reshape(hypos, [1, 32]).numpy()[0], labels.numpy()[0])
            val_accuracy.append(accuracy)
        
        models.append(model)

    tc = np.array([train_cost])
    tc_df = pd.DataFrame(tc.T, columns=["train_cost"])
    tc_df.to_csv("./model_results/{0}_secondary_train_cost.csv".format(name), index=False)

    va = np.array([val_accuracy])
    va_df = pd.DataFrame(va.T, columns=["val_accuracy"])
    va_df.to_csv("./model_results/{0}_secondary_validation_accuracy.csv".format(name), index=False)