import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from collections import defaultdict

import tensorflow as tf
from tensorflow import keras

loaded = tf.saved_model.load("./trained_model/")
print(list(loaded.signatures.keys()))

df = pd.read_csv("./dataset/test.csv")
df = df.astype(float)
df = df.sample(frac=1).reset_index(drop=True)

response = df.pop('y')
print(response.values)

for d in df.values:
    print(loaded.signatures['serving_default'](tf.convert_to_tensor(d)))
    break