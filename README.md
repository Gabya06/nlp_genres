# Movie Genre Prediction Using Machine Learning & NLP

## Project Overview
This project explores how to predict movie genres using **machine learning and NLP techniques**. By analyzing movie synopses, we aim to determine which genres are most popular at different times of the year and whether we can accurately classify a movie’s genre based on its description.

We explore both **traditional ML models (Naïve Bayes, SVM, Gradient Boosting)** and **deep learning approaches (Universal Sentence Encoder for semantic similarity)** to improve genre predictions.


📖 Read the full blog post on: [medium.com](https://medium.com/@gabya06/predicting-movie-genres-using-machine-learning-models-and-semantic-textual-similarity-f77a5e842a89).


## Table of Contents

* [Data Collection](#collectdata)
* [Data Pre-processing](#dataprocessing)
* [Feature Engineering (TF-IDF)](#tfidf)
* [Model Training & Genre Predictions](#genrepred) 
* [Semantic Textual Similarity](#tf)
* [Results & Insights](#results)


---
<a id="collectdata"></a>
## Data Collection
We use the TMDB API to collect movie descriptions, genres, release dates, and other metadata.
---

<a id="dataprocessing"></a>
## Data Preprocessing
* Tokenization, stopword removal, and cleaning noisy descriptions
* Filtering out incomplete or misleading entries
* Transforming text data into numeric representations using *TF-IDF*

---
<a id="tfids"></a>
Feature Engineering (TF-IDF)
To convert text into numerical features, we use TF-IDF vectorization, which helps emphasize important words while down-weighting common terms.

``` python
from sklearn.feature_extraction.text import TfidfVectorizer
tf_vec = TfidfVectorizer()
X_train = tf_vec.fit_transform(train_corpus)
```
---

<a id="genrepred"></a>
## Model Training & Predictions
We train multiple models to classify movies into genres:
* Naïve Bayes – Baseline model
* SVM (Support Vector Machine) – Improved classification performance
* Gradient Boosting Classifier – Boosted performance by combining multiple models

```python
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)
```

---
<a id="tf"></a>
## Semantic Textual Similarity
We use The Universal Sentence Encoder to embed descriptions into a high-dimensional space, which allows us to compute similarity scores between new movies and our known training data genres.

```python
import tensorflow_hub as hub
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def embed(input):
    return model(input)
```

The genre of a new movie is determined by finding the **most similar** show description in the dataset.


---

<a id="results"></a>
## Results & Insights
* **Final Accuracy**: 80% <br>
* **Key Improvements**: Merging overlapping genres and using ensemble methods led to 5% improvement.
---

