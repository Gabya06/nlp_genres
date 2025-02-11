from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfVec:
    """
    Class used for creating and training TFIDF vector on corpus before training NLP Models
    """

    def __init__(self, X_train):
        self.tf_vec = TfidfVectorizer()
        self.max_features = 1000
        self.ngram_range = (1, 2)
        self.X_train = X_train

    def update_tfidf_params(self, max_features, ngram_range):
        """
        Update TF-IDF Vectorizer features
        :param max_features: int, default is 1000
        :param ngram_range: tuple, default (1,2)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range

    def train_tfidf(self):
        """
        Function to create TFIDF matrix representation of training corpus
        This needs to be done before model training as it is used as input in pipeline

        Note:
            TfIdVec object must have X_train:
            pd.Series with string data to be transformed to TFIDF matrix representation

        Returns: tf_vec, X_train (transformed), dict_vocab_tf

        """
        # Tf-Idf Vectorizer
        self.tf_vec = self.tf_vec.fit(self.X_train)
        # transform corpus to tfdif representation
        X_train_matrix = self.tf_vec.transform(self.X_train)
        return X_train_matrix
