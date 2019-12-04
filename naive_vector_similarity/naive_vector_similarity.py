import os
import pandas
from numpy import dot
from numpy.linalg import norm
import numpy as np

from flair.models import TextClassifier
from flair.data import Sentence


def similarity(a, b):
    return abs(a-b)


data_dir = os.path.join(os.getcwd(), os.path.normpath("../Data"))
data_path = os.path.join(data_dir, "smolData.csv")
test_path = os.path.join(data_dir, "Test.csv")
validation_path = os.path.join(data_dir, "Validation.csv")
output_path = os.path.join(os.getcwd(), "prediction.txt")
gold_path = os.path.join(data_dir, "gold.txt")


#train_df = pandas.read_csv(data_path)
test_df = pandas.read_csv(validation_path)
# val_df = pandas.read_csv(test_path)

#train = {t[0]: t[1:] for t in train_df.values.tolist()}
test = {t[0]: t[1:] for t in test_df.values.tolist()}
# val = {t[0]: t[1:] for t in val_df.values.tolist()}

lines = list(test.values())
#lines = list(train.values())

expected = []
predicted = []

classifier = TextClassifier.load('en-sentiment')
for inp in lines:
    sum = 0
    for i in range(0, 4):
        s = Sentence(inp[i])
        classifier.predict(s)
        sum += s.labels[0].score;
    s = Sentence(inp[4])
    classifier.predict(s)
    one_sent = s.labels[0].score

    s = Sentence(inp[5])
    classifier.predict(s)
    two_sent = s.labels[0].score

    predicted.append("1" if similarity(sum/4,one_sent) > similarity(sum/4,  two_sent) else "2")
    expected.append(str(inp[6]))

with open(output_path, 'w') as f:
    f.write("\n".join(predicted))

with open(gold_path, 'w') as f:
    f.write("\n".join(expected))

    #flair citation

#@inproceedings{akbik2018coling,
#  title={Contextual String Embeddings for Sequence Labeling},
#  author={Akbik, Alan and Blythe, Duncan and Vollgraf, Roland},
#  booktitle = {{COLING} 2018, 27th International Conference on Computational Linguistics},
#  pages     = {1638--1649},
#  year      = {2018}
