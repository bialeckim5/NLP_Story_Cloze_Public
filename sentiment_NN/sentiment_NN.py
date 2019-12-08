import os
import pandas
from numpy import dot
from numpy.linalg import norm
import numpy as np
from sklearn.neural_network import MLPClassifier
import ast



data_dir = os.path.join(os.getcwd(), os.path.normpath("../Data"))
data_path = os.path.join(data_dir, "Data_sentiment.csv")
test_path = os.path.join(data_dir, "Test_sentiment.csv")
validation_path = os.path.join(data_dir, "Validation_sentiment.csv")
output_path = os.path.join(os.getcwd(), "prediction.txt")
gold_path = os.path.join(data_dir, "gold.txt")

def get_compound(x):
    return ast.literal_eval(x)['compound']

train_df = pandas.read_csv(data_path)
test_df = pandas.read_csv(validation_path)
# val_df = pandas.read_csv(test_path)

train_expected = train_df['AnswerRightEnding']
test_expected = test_df['AnswerRightEnding']

train_df = train_df.drop('AnswerRightEnding', 1)
test_df = test_df.drop('AnswerRightEnding', 1)

train_df = train_df.drop('InputStoryid', 1)
test_df = test_df.drop('InputStoryid', 1)

train_df = train_df.drop('Unnamed: 0', 1)
test_df = test_df.drop('Unnamed: 0', 1)

train_df = train_df.applymap(get_compound)
test_df = test_df.applymap(get_compound)

model = MLPClassifier()
model.fit(train_df, train_expected)
predicted = model.predict(test_df)

np.savetxt(output_path,predicted, delimiter=",", fmt='%i')
test_expected.to_csv(gold_path, header=False, index=False)

