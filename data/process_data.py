import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    print(messages.head())
    print('Shape of messages data set: ', messages.shape)
    categories = pd.read_csv(categories_filepath)
    print(categories.head())
    print('Shape of categories data set: ', categories.shape)

    merged_df = merge_datasets(messages, categories)

    return merged_df
    

def merge_datasets(df1, df2):
    merged_df = df1.merge(df2, on ='id')
    #print(merged_df.head())
    print('Shape of merged data set: ', merged_df.shape)
    return merged_df


def clean_data(df):
    df_split = df['categories'].str.split(';', expand=True).add_prefix('name_')
    print('Shape of split data set: ', df_split.shape)
    # get first cloumn
    row = df_split.iloc[[0]]
    # get all but not the last 2 string characters from row values
    category_colnames = row.apply(lambda x: x.str[:-2], axis=1)
    # create list of new columns values set data frame
    df_split.columns = category_colnames.iloc[0]
    # set each value to be the last character of the string and convert column from string to numeric
    df_split = df_split.apply(lambda x: np.int8(x.str[-1:]), axis=0)
    ## drop original categories
    df = df.drop(columns=['categories'])
    new_df = pd.concat([df, df_split] ,axis=1)
    print(new_df.head())
    

    # Select duplicate rows except first occurrence based on all columns
    duplicateRowsDF = new_df[new_df.duplicated()]
    
    print("Duplicate Rows except first occurrence based on all columns are :")
    print(duplicateRowsDF)

    result_df = new_df.drop_duplicates(keep='first')
    print(result_df.shape)

        # Select duplicate rows except first occurrence based on all columns
    duplicateRowsDF = result_df[result_df.duplicated()]
    
    print("Remaining duplicates after cleaning: ")
    print(len(duplicateRowsDF))

    return result_df


    


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename+'.db')
    df.to_sql('messages_and_categories', engine, index=True)



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
    # categories_filepath = 'disaster_categories.csv'
    # messages_filepath = 'disaster_messages.csv'
    # df = load_data(messages_filepath, categories_filepath)
    # df = clean_data(df)
    # save_data(df, 'database_filename')
    main()