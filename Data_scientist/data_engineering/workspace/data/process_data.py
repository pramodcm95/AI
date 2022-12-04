import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function loads two files in csv format and merges them
    Input:
    messages_filepath - path to messages.csv file
    messages_filepath - path to categories.csv file
    
    Returns:
    Dataframe with messages and categories merged
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    This function cleans the given data by general pre processing ideas like removing duplicates, removing invalid data and others
    Input:
    df - Raw dataframe
    
    Returns:
    Cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[1]

    # use this row to extract a list of new column names for categories.
    categories.columns = row
    category_colnames = row.apply(lambda item : item[:len(item) - 2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Checking for non-binay nature of any categories
    columns_to_be_processed = []
    for column in categories:
        if len(categories[column].unique()) >= 3:
            columns_to_be_processed.append(column)
            
    # using value_counts to decide what to do with this column
    for col in columns_to_be_processed:
        print(categories[col].value_counts())

    # we are renaming related-2 as related-1
    categories[columns_to_be_processed[0]] = categories[columns_to_be_processed[0]].apply(lambda x: categories[columns_to_be_processed[0]].unique()[1] if x==categories[columns_to_be_processed[0]].unique()[2] else x)  
    
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda item:  item[len(item)-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        categories = categories.rename_axis(None, axis=1)

    # drop the original categories column from `df`
    df = df.drop('categories',axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)
    
    # drop duplicates
    df = df.drop_duplicates(keep=False)
    
    return df
    
def save_data(df, database_filename):
    """
    This function saves the processed data from ETL pipeline to Database
    Input:
    df - Processed dataframe
    database_filename - Path to db

    """
    engine = create_engine('sqlite:///'+database_filename.split('/')[1])
    df.to_sql(database_filename.split('/')[1].split('.')[0], engine, index=False, if_exists='replace')  
    print("---- Data is clean and Ready for modelling ----")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()