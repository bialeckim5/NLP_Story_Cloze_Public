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

lines =0
counter = 0
def get_compound(x):
    if (x.count() >lines):
        return x
    for (indx,val) in enumerate(x.iteritems()):
        result = ast.literal_eval(val[1])
        x[val[0] + "pos"] = result['pos']
        x[val[0] +"neg"] = result['neg']
        x[val[0] + "neu"] = result['neu']
        global counter
        counter +=1
        print(counter)
    return x

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

lines = train_df.shape[0]
train_df = train_df.apply(get_compound, axis = 1)
train_df = train_df.drop('InputSentence1', 1)
train_df = train_df.drop('InputSentence2', 1)
train_df = train_df.drop('InputSentence3', 1)
train_df = train_df.drop('InputSentence4', 1)
train_df = train_df.drop('RandomFifthSentenceQuiz1', 1)
train_df = train_df.drop('RandomFifthSentenceQuiz2', 1)

lines = test_df.shape[0]
test_df = test_df.apply(get_compound, axis = 1)
test_df = test_df.drop('InputSentence1', 1)
test_df = test_df.drop('InputSentence2', 1)
test_df = test_df.drop('InputSentence3', 1)
test_df = test_df.drop('InputSentence4', 1)
test_df = test_df.drop('RandomFifthSentenceQuiz1', 1)
test_df = test_df.drop('RandomFifthSentenceQuiz2', 1)


model = MLPClassifier()
model.fit(train_df, train_expected)
predicted = model.predict(test_df)

np.savetxt(output_path,predicted, delimiter=",", fmt='%i')
test_expected.to_csv(gold_path, header=False, index=False)

