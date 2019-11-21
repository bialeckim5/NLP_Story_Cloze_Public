## Results

| Model | Accuracy | Macro Precision | Macro Recall |
| --- | --- | --- | --- |
| Baseline | 0.51042 | 0.25521 | 0.5 |
| Naive vector similarity | 0.58311 | 0.58301 | 0.58304 |

#### Baseline

Our baseline for performance comparison is a simple majority vote.

#### Naive vector similarity

For this first approach, the four context sentences, first and second conclusions are vectorized using spaCy embeddings, specifically from the `en_core_web_md` model.  The conclusion with the closest (largest) cosine similarity to the context is the predicted answer. 

 ---
### Data

The supporting data for this project must be requested manually:

https://www.cs.rochester.edu/nlp/rocstories/

The data for this project (supplied by Univ. of Rochester CS) will not be shared outside of the above source.

Data should be stored in the `Data` directory, which is ignored in git.  You must download the dataset and create that folder yourself locally.