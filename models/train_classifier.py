import sys
import pandas as pd
import numpy as np
import re
import nltk
import pickle
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP']:
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

class StartingNounExtractor(BaseEstimator, TransformerMixin):
    
    def starting_noun(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag == 'NN':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_noun)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    """Loads data from data base and prepares input and output for classification
    
    Arguments:
        database_filepath {string} -- Name of the database
    
    Returns:
        panda Dataframes -- features, target variable, category_names
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('table_A', engine)
    #print(df.head())
    #print(df.columns)

    X = df['message']
    ## drop child_alone because it contains only 0
    Y = df.drop(columns=['message','original', 'genre','index','id', 'child_alone'])
        # Y['related'] contains three distinct values
    # mapping extra values to `1`
    Y['related']=Y['related'].map(lambda x: 1 if x == 2 else x)

    for col in Y:
        print(col, Y[col].unique())
    
    category_names = Y.columns

    return X, Y, category_names





def tokenize(text):
    """Tokenizes text data
    
    Arguments:
        text {string} -- Messages as text data
    
    Returns:
        words list: Processed text after normalizing, tokenizing and lemmatizing
    """


    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)

    unique_words = [w for w in tokens if w not in stopwords.words("english")]

    # Reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in unique_words]

    # Lemmatize verbs by specifying pos
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in stemmed]

    return lemmed


def build_model():

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor()),
            ('starting_noun', StartingNounExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier())) #LinearSVC()
    ])

    # hyper-parameter grid
    parameters = {
                    'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
                  'features__text_pipeline__vect__max_df': (0.75, 1.0),
                  'clf__estimator__n_estimators': (10, 50)
                  }

    # create model
    model = GridSearchCV(estimator=pipeline,
            param_grid=parameters,
            verbose=2,
            cv=3)

    return model




def evaluate_model(model, X_test, y_test, category_names, model_filepath):
    y_pred = model.predict(X_test)
    #labels = np.unique(y_pred)
    #confusion_mat = confusion_matrix(y_test, np.argmax(y_pred, axis=1), labels=category_names)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", category_names)
    #print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)

    print("Classification report: ")
    print(classification_report(y_test, y_pred, target_names=category_names))
    report = classification_report(y_test, y_pred, target_names=category_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    try:
        df.to_json(model_filepath+'.json')
    except:
        print('Failed to save classification report!')


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)




def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    # database_filepath = './data/database_test.db'
    # load_data(database_filepath)
    # build_model()
    #python train_classifier.py ../data/database_test.db model_pickle.pkl
    main()