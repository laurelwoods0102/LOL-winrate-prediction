import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    #dataframe = dataframe.copy()
    response = dataframe.pop('result')    
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), response.values))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    dataset = dataset.batch(batch_size)

    return dataset

def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

class LogisticLayer(keras.layers.Layer):
    def __init__(self, input_dim=146, num_outputs=1):
        super(LogisticLayer, self).__init__()
        w_init = tf.initializers.GlorotUniform()     # NOTICE : weight matrix itself contains bias
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, num_outputs), dtype='float32'),
            trainable=True)
        #print(self.w)

    def call(self, features):
        return tf.matmul(features, self.w)

class LogisticModel(tf.keras.Model):
    def __init__(self):
        super(LogisticModel, self).__init__()
        self.logistic_layer = LogisticLayer()
    
    def call(self, features, training=False):
        x = self.logistic_layer(features)
        return x

def hypothesis(logit):
    return tf.divide(1.0, 1.0 + tf.exp(-1.0*logit))

def cost(logits, labels):
    hypo = hypothesis(logits)
    #return -tf.reduce_mean(labels*tf.math.log(hypo) + (1.0-labels)*tf.math.log(1.0-hypo))
    return -tf.reduce_mean(tf.math.log(hypo)*labels + tf.math.log(1.0-hypo)*(1.0-labels))

def grad(logits, labels):
    with tf.GradientTape() as tape:
        cost_value = cost(model(features), labels)
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

if __name__ == "__main__":
    df = pd.read_csv("./dataset/dataset_hide-on-bush_enemy.csv")
    df = df.astype(float)
    df = df.sample(frac=1).reset_index(drop=True)   # shuffle
    
    train_len, test_len = divmod(len(df), 32)
    if test_len == 0:
        train_len -= 32
        test_len = 32
    
    
    

'''
if __name__ == "__main__":
    df = pd.read_csv("./dataset/dataset_hide-on-bush_enemy.csv")
    df = df.astype(float)

    train_df, test_df = train_test_split(df, test_size=0.1)
    train_df, val_df = train_test_split(train_df, test_size=0.1)


    train_ds = df_to_dataset(train_df, shuffle=True)
    val_ds = df_to_dataset(val_df, shuffle=False)
    test_ds = df_to_dataset(test_df, shuffle=False)


    train_dataset = train_ds.map(pack_features_vector)
    #for features, labels in train_dataset.take(1):
    #    print(features)

    ex_ds = df_to_dataset(df, shuffle=True)
    ex_dataset = ex_ds.map(pack_features_vector)


    model = LogisticModel()
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    f = open('./model_results/version_1/cost.csv', 'w', newline='')
    w = open('./model_results/version_1/weights.csv', 'w', newline='')
    accu = open('./model_results/version_1/accuracy.txt', 'w')
    cost_record = open('./model_results/version_1/cost.txt', 'w')

    Ws = list()
    train_cost_results = list()
    train_accuracy_results = list()

    epochs = range(25)
    for epoch in epochs:
        print('epoch : ', epoch)
        temp_accuracy = list()
        temp_cost = list()
        for features, labels in ex_dataset:
            labels = tf.dtypes.cast(labels, tf.float32) # reshape Labels
            labels = tf.reshape(labels, [32, 1])
            labels = tf.transpose(labels)

            current_cost = train(model, features, labels)

            logits = model(features)
            hypos = hypothesis(logits)

            accuracy = batch_accuracy(tf.reshape(hypos, [1, 32]).numpy()[0], labels.numpy()[0])
            
            accu.write(str(accuracy))
            accu.write("\n")
            cost_record.write(str(current_cost.numpy()))
            cost_record.write("\n")

            temp_accuracy.append(accuracy)
            temp_cost.append(current_cost.numpy())
        
        train_accuracy_results.extend(temp_accuracy)
        train_cost_results.extend(temp_cost)
'''