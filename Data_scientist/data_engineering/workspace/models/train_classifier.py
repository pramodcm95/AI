import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
import sklearn
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score,accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import pickle 

def load_data(database_filepath):
    """
    This function loads the data from ETL pipeline
    Input:
    database_filepath - Path to db with ETL processed data
    
    Returns:
    X - Entire dataset with features only
    y - Categories each datapoint belong to
    Categories - Different categories avilable in dataset
    """
    # load data from databasepython app
    engine = create_engine('sqlite:///'+database_filepath.split('/')[1])
    df = pd.read_sql_table(database_filepath.split('/')[1].split('.')[0],engine)
    # dropping a column which is just traslation in different language
    df = df.drop('original',axis=1)
    # there are some rows with empty sapces as messages which will thrw an error letter
    df['message'] = df['message'].str.strip()
    # Removing all such rows
    df = df[df['message'] != '']
    X = df[df.columns[1:2]].values
    y = df[df.columns[3:]].values
    return X, y, list(df.columns[3:])

def tokenize(text):
    """
    This function takes a raw text and returns cleaned tokens
    Input:
    text - Raw text(messages)
    
    Returns:
    clean_tokens - A list of individual words in a given sentence
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    #print(text)
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    #if len(clean_tokens) == 0:
        #print('True')
    return clean_tokens

def build_model():
    """
    This function builds the machine learning pipeline
    
    Returns:
    pipeline - Sklearn ML pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('model',MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'model__estimator__n_estimators': [5, 10, 20],
        'model__estimator__min_samples_split': [0.1,0.2,0.4],
        #'model__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates the trained pipeline
    Input:
    model - Trained model
    X_test - Test set features
    Y_test - Categories of each test datapoint
    category_names - all possible categories name in dataset
    """
    y_pred = model.predict(X_test)
    for col in range(Y_test.shape[1]):
        print("Current Category is titles  "+ category_names[col] )
        print(classification_report(Y_test[:,col] ,y_pred[:,col]))

def save_model(model, model_filepath):
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
        model.fit(X_train.reshape(-1), Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test.reshape(-1), Y_test, category_names)

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