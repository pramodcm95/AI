# Disaster Response Pipeline Project
"""
Introduction:

In this project titled "Disaster response Pipeline", I have considered raw messages as inputs, and build a data pipeline to pre-process the data(with respect to NLP) and clean it wherever necessary, so that the data is ready to be fed into the models/machine learning pipeline. The machine learning pipeline tries to classify the given messages to one or more of the possible categories(sum total of 36 categories exist here). The result consists of F1_score, precision, recall, acuuracy, i.e. classification report in general, for each of the 36 categories and model seems to perform decently. Later taking a step forward, instead of normal testing, a web application has been built where user can give their own message which our classifier process it and assigns it respective category/categories
"""
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
