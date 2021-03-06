import os
import pandas
import spacy
from keras import Sequential
from keras.engine.saving import model_from_json
from keras.layers import LSTM, Dense, Dropout
from keras.utils import plot_model
from keras.regularizers import l2
from numpy import dot
from numpy.linalg import norm
import numpy as np

USE_PREPROCESSED = True
FRESH_MODEL = False
MODEL_NAME = 'dropout_validation_5473'


def build_model():
    model = Sequential()
    model.add(Dense(400, input_shape=(1500,), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, x, y, test_x, test_y, epochs=3):
    model.fit(x, y, batch_size=64, epochs=epochs, validation_data=(test_x, test_y))


def evaluate_model(model, x, y):
    return model.evaluate(x, y)


def image_model(model, name):
    plot_model(model, to_file=name, show_shapes=True, show_layer_names=True)


def save_model(model, name):
    with open(name + ".json", 'w') as f:
        f.write(model.to_json())

    model.save_weights(name + ".h5")


def load_model_from_file(name):
    with open(name + ".json", 'r') as f:
        model = model_from_json(f.read())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.load_weights(name + ".h5")
    return model

def similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


data_dir = os.path.join(os.getcwd(), os.path.normpath("../Data"))
data_path = os.path.join(data_dir, "Data.csv")
test_path = os.path.join(data_dir, "Test.csv")
validation_path = os.path.join(data_dir, "Validation.csv")
output_path = os.path.join(os.getcwd(), "prediction.txt")
gold_path = os.path.join(data_dir, "gold.txt")


if not USE_PREPROCESSED:

    train_df = pandas.read_csv(data_path)
    test_df = pandas.read_csv(validation_path)
    val_df = pandas.read_csv(test_path)

    train = {t[0]: t[1:] for t in train_df.values.tolist()}
    test = {t[0]: t[1:] for t in test_df.values.tolist()}
    val = {t[0]: t[1:] for t in val_df.values.tolist()}

    sp = spacy.load("en_core_web_md")

    lines_test = list(test.values())
    lines_train = list(train.values())

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    expected = []
    predicted = []

    for inp in lines_test:
        input_vec = [sp(inp[i]).vector for i in range(0, 4)]
        one_vec = sp(inp[4]).vector
        two_vec = sp(inp[5]).vector
        x_test.append(input_vec + [one_vec])
        y_test.append(1 if inp[6] == 1 else 0)

        x_test.append(input_vec + [two_vec])
        y_test.append(1 if inp[6] == 2 else 0)

    counter = 0
    for inp in lines_train:
        input_vec = [sp(inp[i]).vector for i in range(0, 4)]
        one_vec = sp(inp[4]).vector
        two_vec = sp(inp[5]).vector

        x_train.append(input_vec + [one_vec])
        y_train.append(1 if inp[6] == 1 else 0)

        x_train.append(input_vec + [two_vec])
        y_train.append(1 if inp[6] == 2 else 0)

        counter = counter + 1
        if counter % 1000 == 0:
            print("Read", int(counter / 1000))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_test = np.reshape(x_test, (len(x_test), 1500))
    x_train = np.reshape(x_train, (len(x_train), 1500))

    with open('x_test', 'wb') as f:
        np.save(f, x_test)
    with open('y_test', 'wb') as f:
        np.save(f, y_test)
    with open('x_train', 'wb') as f:
        np.save(f, x_train)
    with open('y_train', 'wb') as f:
        np.save(f, y_train)

else:

    with open('x_test', 'rb') as f:
        x_test = np.load(f)
    with open('y_test', 'rb') as f:
        y_test = np.load(f)
    with open('x_train', 'rb') as f:
        x_train = np.load(f)
    with open('y_train', 'rb') as f:
        y_train = np.load(f)

if FRESH_MODEL:
    model = build_model()

    train_model(model, x_train, y_train, x_test, y_test, 10)
    print(evaluate_model(model, x_test, y_test))
    save_model(model, MODEL_NAME)
else:
    model = load_model_from_file(MODEL_NAME)

    # print('train', evaluate_model(model, x_train, y_train))
    print('test', evaluate_model(model, x_test, y_test))

predictions = model.predict(x_test)
predicted = ["1" if p >= 0.5 else "0" for p in predictions]
expected = ["1" if y == 1 else "0" for y in y_test]

with open(output_path, 'w') as f:
    f.write("\n".join(predicted))

with open(gold_path, 'w') as f:
    f.write("\n".join(expected))
