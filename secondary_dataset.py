import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from collections import defaultdict

import tensorflow as tf
from tensorflow import keras



class LogisticLayer(keras.layers.Layer):
    def __init__(self, weights):
        super(LogisticLayer, self).__init__()
        self.w = tf.convert_to_tensor(weights)

    @tf.function
    def call(self, features):
        return tf.matmul(features, self.w)

class LogisticModel(tf.keras.Model):
    def __init__(self, weights):
        super(LogisticModel, self).__init__()
        self.logistic_layer = LogisticLayer(weights)    #input shape = 146, 4

    @tf.function
    def call(self, features, training=False):
        x = self.logistic_layer(features)
        return x

def hypothesis(logit):
    return tf.divide(1.0, 1.0 + tf.exp(-1.0*logit))

def processor(name, model_type):
    name = name.replace(' ', '-')
    weights = np.load('./trained_model/weights_{0}_{1}.npy'.format(name, model_type))


    df = pd.read_csv("./dataset/dataset_{0}_{1}.csv".format(name, model_type))
    df = df.astype(float)
    response = df.pop('y')
    
    model = LogisticModel(weights)

    predictions = list()
    for data in df.values:
        data = np.reshape(data, (1, 146))        
        logit = model(tf.convert_to_tensor(data))
        predict = hypothesis(logit).numpy()[0][0]
        predictions.append(predict)
    
    data = np.array([predictions])
    dataset = pd.DataFrame(data.T, columns=["predict_{}".format(model_type)])
    dataset.to_csv("./dataset/secondary/dataset_2_{0}_{1}.csv".format(name, model_type), index=False, header=False)
    
    response.to_csv("./dataset/secondary/dataset_2_response.csv", index=False, header=False)


if __name__ == "__main__":
    processor("hide on bush", "enemy")