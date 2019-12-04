import os
import pandas
import flair
from numpy import dot
from numpy.linalg import norm
import numpy as np




def similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


sentiment_classifier = flair.models.TextClassifier.load('en-sentiment')

data_dir = os.path.join(os.getcwd(), os.path.normpath("../Data"))
data_path = os.path.join(data_dir, "Data.csv")
test_path = os.path.join(data_dir, "Test.csv")
validation_path = os.path.join(data_dir, "Validation.csv")
output_path = os.path.join(os.getcwd(), "prediction.txt")
gold_path = os.path.join(data_dir, "gold.txt")


train_df = pandas.read_csv(data_path)
test_df = pandas.read_csv(validation_path)
val_df = pandas.read_csv(test_path)

train = {t[0]: t[1:] for t in train_df.values.tolist()}
test = {t[0]: t[1:] for t in test_df.values.tolist()}
val = {t[0]: t[1:] for t in val_df.values.tolist()}

lines = list(test.values())

expected = []
predicted = []

for inp in lines:
    sum = 0rtwo_vec = sp(inp[5]).vector

    predicted.append("1" if similarity(one_vec, input_vec) > similarity(two_vec, input_vec) else "2")
    expected.append(str(inp[6]))

with open(output_path, 'w') as f:
    f.write("\n".join(predicted))

with open(gold_path, 'w') as f:
    f.write("\n".join(expected))
