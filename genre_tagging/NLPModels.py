from joblib import dump, load
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def get_score(y_true, y_pred):
    """
    Return accuracy score between actual vs predicted values
    :param y_true:
    :param y_pred:
    :return:model accuracy score
    """
    return accuracy_score(y_true, y_pred)


class NLPModels:
    """
    Class for NLP Models:
        Can pass in a classification model (e.g: Naive Bayes, Logistic Regression, SVM...)
        Train model: pass in TfIdf trained model, create pipeline  & fit model
        Tune model parameters: perform 3fold cross-validation and get best params
        Predict: make model predictions on test data
    """

    def __init__(self, model, model_name):
        self.model = model
        self.model.random_state = 23
        self.model_name = model_name

    def update_model(self, new_model):
        """
        Set new model

        :param new_model: classification model
        """
        self.model = new_model

    def set_model_params(self, **params):
        """
        Update model parameters:
            Called after tune_params runs to update model params based on cross-validation results

        :param params: dict with best params
        """
        self.model.set_params(**params)

    def tune_params(self, X_train, y_train, **grid_params):
        """
        Perform  cross-validation to find optimal model parameters & set as model parameters

        :param X_train: Training data. Should be TfIDF matrix
        :param y_train: Labels
        :param grid_params: dict with params to search through
            (e.g: sv_losses = ['hinge', 'log', 'modified_huber'] alphas = [0.1, 0.01, 0.05, 0.001]
                    param_search = {'loss': sv_losses, 'alpha': alphas}
        """
        tuned_model = GridSearchCV(self.model, grid_params, cv=3)
        tuned_model.fit(X_train, y_train)
        best_params = tuned_model.best_params_
        best_score = tuned_model.best_score_
        self.set_model_params(**best_params)
        print(
            f"Finished grid search... best params {best_params} best score {best_score}"
        )

    def train_model(self, X_train, y_train, **model_params):
        """
        Train model
        * Note:
            if pass in TfIDF then create lst with [TfIDF, model] and pass to Pipeline,
            otherwise just use [model] in Pipeline

        :param X_train: TfIdf matrix representation of training corpus
        :param y_train: labels
        :param model_params: dict used to pass in trained TfIdf vectorizer.

            e.g: tf_vec = TfIdfVec.TfIdfVec(X_train=X)
                v = {'vectorizer': tf_vec.tf_vec}
                X_matrix = tf_vec.train_tfidf()
                nb = NLPModels.NLPModels(model=MultinomialNB(), model_name='NB')
                nb.train_model(X_matrix, y, **v)
        """
        # list with model
        model_lst = [("classifier", self.model)]

        # pipeline & classification model
        if model_params:
            lst = list(model_params.items())
            lst.extend(model_lst)
            model = Pipeline(lst)
        else:
            model = Pipeline(model_lst)
        # train classifier
        model["classifier"].fit(X_train, y_train)
        self.model = model

    def predict(self, X_test):
        """
        Make predictions on test data

        :param X_test: DataFrame with test data
        :return: model predictions (array)
        """
        return self.model.predict(X_test)

    def export_model(self):
        """
        Export model using joblib library
        """
        filename = self.model_name + ".joblib"
        dump(self.model, filename)
        print(f"Saved model {self.model_name} to {filename}")

    def load_model(self, filename):
        """
        Load previously trained & exported model as joblib file (filename must end in '.joblib')

        :param filename:str, model filename to load. eg: NB.joblb
        """
        # loaded_model = pickle.load(open(filename, 'rb'))
        loaded_model = load(filename)
        self.model = loaded_model
        print(f"Loaded model {self.model_name} from {filename}")
