import os
import pandas
import spacy
from numpy import dot
from numpy.linalg import norm
import numpy as np


def similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


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

sp = spacy.load("en_core_web_md")

lines = list(val.values())

expected = []
predicted = []

for inp in lines:
    input_vec = np.mean([sp(inp[i]).vector for i in range(0, 4)], axis=0)
    one_vec = sp(inp[4]).vector
    two_vec = sp(inp[5]).vector

    predicted.append("1" if similarity(one_vec, input_vec) > similarity(two_vec, input_vec) else "2")
    # expected.append(str(inp[6]))

with open(output_path, 'w') as f:
    f.write("\n".join(predicted))

# with open(gold_path, 'w') as f:
#     f.write("\n".join(expected))
