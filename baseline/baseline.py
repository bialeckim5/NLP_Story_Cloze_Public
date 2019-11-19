import os
import pandas

data_dir = os.path.join(os.getcwd(), os.path.normpath("../Data"))
data_path = os.path.join(data_dir, "Data.csv")
test_path = os.path.join(data_dir, "Test.csv")
validation_path = os.path.join(data_dir, "Validation.csv")
output_path = os.path.join(os.getcwd(), "prediction.txt")
gold_path = os.path.join(data_dir, "gold.txt")

train_df = pandas.read_csv(data_path)
val_df = pandas.read_csv(test_path)
test_df = pandas.read_csv(validation_path)

answer_counts = train_df.groupby('AnswerRightEnding')['AnswerRightEnding'].agg(['count'])
test_answer_counts = test_df.groupby('AnswerRightEnding')['AnswerRightEnding'].agg(['count'])
max_level = (answer_counts.idxmax())[0]
# max_count = test_answer_counts.loc[max_level][0]
all_count = test_df.shape[0]

test_predicted = [str(max_level)] * all_count
test_expected = list(test_df['AnswerRightEnding'].astype(str))

with open(output_path, 'w') as f:
    f.write("\n".join(test_predicted))

with open(gold_path, 'w') as f:
    f.write("\n".join(test_expected))
