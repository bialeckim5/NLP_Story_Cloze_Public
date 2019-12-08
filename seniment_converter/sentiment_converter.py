import os
import sys

import pandas

from flair.models import TextClassifier
from flair.data import Sentence



data_dir = os.path.join(os.getcwd(), os.path.normpath("../Data"))
input_path = os.path.join(data_dir, sys.argv[1])
output_path = os.path.join(data_dir, sys.argv[1] + "sentiment")


input_df = pandas.read_csv(input_path)
classifier = TextClassifier.load('en-sentiment')

def sentiment(input):
    s = Sentence(input)
    classifier.predict(s)
    return str(s.labels[0].score)


for index,row in input_df.iterrows():

    input_df.loc[index, 'InputSentence1'] = sentiment(row['InputSentence1'])
    input_df.loc[index, 'InputSentence2'] = sentiment(row['InputSentence2'])
    input_df.loc[index, 'InputSentence3'] = sentiment(row['InputSentence3'])
    input_df.loc[index, 'InputSentence4']  = sentiment(row['InputSentence4'])
    input_df.loc[index, 'RandomFifthSentenceQuiz1'] = sentiment(row['RandomFifthSentenceQuiz1'])
    input_df.loc[index, 'RandomFifthSentenceQuiz2'] = sentiment(row['RandomFifthSentenceQuiz2'])
    print("input")


input_df.to_csv(output_path, sep=',')


    #flair citation

#@inproceedings{akbik2018coling,
#  title={Contextual String Embeddings for Sequence Labeling},
#  author={Akbik, Alan and Blythe, Duncan and Vollgraf, Roland},
#  booktitle = {{COLING} 2018, 27th International Conference on Computational Linguistics},
#  pages     = {1638--1649},
#  year      = {2018}
