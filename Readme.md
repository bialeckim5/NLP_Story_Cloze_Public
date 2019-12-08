## Results

| Model | Accuracy | Macro Precision | Macro Recall |
| --- | --- | --- | --- |
| Baseline | 0.51042 | 0.25521 | 0.5 |
| Naive vector similarity | 0.58311 | 0.58301 | 0.58304 |
| Basic LSTM on sentence vectors | 0.50508 | 0.51137 | 0.50508 |
| Flair Naive Sentiment vector <not going to turn in>| 0.49653| 0.49644| 0.49644
| NLTK NAIVE SENTIMENT|  0.46392 | 0.46456 | 0.46606
| NTLK LOG REGRESSION (SENTIMENT ONLY) | 0.44896 | 0.44897 | 0.44895
| NTLK NN Classifier | 0.60609 |0.61236 | 0.60315
| SKLearn NN on sentiment | 0.60075 | 0.60074 | 0.60079 |
| NN on spacy vector representation | 0.54730 | 0.57570 | 0.54730 | 

#### Baseline

Our baseline for performance comparison is a simple majority vote.

#### Naive vector similarity

For this first approach, the four context sentences, first and second conclusions are vectorized using spaCy embeddings, specifically from the `en_core_web_md` model.  The conclusion with the closest (largest) cosine similarity to the context is the predicted answer. 


#### Basic LSTM on sentence vectors

For this approach, we take the vectors for the four context sentences (separately), with a vector for the closure, and feed these through an LSTM and learn a binary classification output for whether this is the correct closure.

We first preprocess all data, turning it into 5 sentences (the 4 context sentences with one of the closures), and set the expected label to be 1 if this closure is correct, or 0 if the other is correct.  We do this for both closure outputs, effectively yielding double the training data points.

The model is a very basic one - 5 timeslice points into a keras LSTM layer with a single output.  This output is put through a sigmoid function (Dense layer with 1 output).

![Basic LSTM model](lstm/basic_model.png)


#### NN on spacy vector representation

We trained a keras multi-layer NN comprised of Dense layers on the SpaCy vector representation of the  

 ---
### Data

The supporting data for this project must be requested manually:

https://www.cs.rochester.edu/nlp/rocstories/

The data for this project (supplied by Univ. of Rochester CS) will not be shared outside of the above source.

Data should be stored in the `Data` directory, which is ignored in git.  You must download the dataset and create that folder yourself locally.