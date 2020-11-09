# import libraries
# General
import sys
import pandas as pd
import re
from sqlalchemy import create_engine
import pickle
# Natural Languages
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
# Machine Learning
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

stopeng = stopwords.words('english')

def load_data(database_filepath):
    '''
    load data from database
    input:
    - database filepath
    output:
    - X text column
    - Y target classification
    - col_names category names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("disaster_table", con=engine)
    col_names = df.columns[5:]
    X = df.message.values
    Y = df[col_names].values
    return X, Y, col_names

def tokenize(text):
    '''
    tokenize, lemmatization and delete stopwords for Machine Learning use.
    input:
    - text file
    output:
    - a post-processed list of words
    '''
    # lowercase and remove punctuation
    text = re.sub(r'[^A-Za-z0-9]', ' ', text.lower())
    # tokenize
    words = word_tokenize(text)
    # lemmatize and remove stopwords
    result = [WordNetLemmatizer().lemmatize(w) for w in words if w not in stopeng]
    return result


def build_model():
    '''
    function to bulid a ML Pipeline
    no input
    output:
    - model classifier
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-2, n_estimators=10))),
    ])
    # GridSearchCV
    parameters = {'vect__max_df': (0.5, 0.75, 1.0),
                  'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_features': (None, 5000, 10000),
                  'tfidf__use_idf': (True, False)
                  }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-2, verbose=10)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    function to report f1 score, precision and recall for each output category of the dataset
    input:
    - model trained classifier
    - X_test
    - Y_test
    - category_names
    output:
    - detailed evaluation of each categories
    '''
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    function to save the model
    input:
    - model trained classifier
    - model_filepath
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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
    main()
