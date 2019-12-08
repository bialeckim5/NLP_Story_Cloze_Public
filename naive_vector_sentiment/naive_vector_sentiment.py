import os
import pandas
import ast
from numpy import dot
from numpy.linalg import norm
import numpy as np

def similarity(input1, input2):
    return abs(input1 - input2)



data_dir = os.path.join(os.getcwd(), os.path.normpath("../Data"))
#data_path = os.path.join(data_dir, "Data_sentiment.csv")
test_path = os.path.join(data_dir, "Test_sentiment.csv")
validation_path = os.path.join(data_dir, "Validation_sentiment.csv")
output_path = os.path.join(os.getcwd(), "prediction.txt")
gold_path = os.path.join(data_dir, "gold.txt")


#train_df = pandas.read_csv(data_path)
test_df = pandas.read_csv(validation_path)
# val_df = pandas.read_csv(test_path)

expected = []
predicted = []

for index, row in test_df.iterrows():
    sum = 0
    sum += ast.literal_eval(row['InputSentence1'])['compound']
    sum += ast.literal_eval(row['InputSentence2'])['compound']
    sum += ast.literal_eval(row['InputSentence3'])['compound']
    sum += ast.literal_eval(row['InputSentence4'])['compound']
    average = sum/4
    one_sim = ast.literal_eval(row['RandomFifthSentenceQuiz1'])['compound']
    two_sim = ast.literal_eval(row['RandomFifthSentenceQuiz2'])['compound']
    predicted.append("1" if similarity(average,one_sim) > similarity(average,  two_sim) else "2")
    expected.append(str(row['AnswerRightEnding']))


with open(output_path, 'w') as f:
    f.write("\n".join(predicted))

with open(gold_path, 'w') as f:
    f.write("\n".join(expected))

