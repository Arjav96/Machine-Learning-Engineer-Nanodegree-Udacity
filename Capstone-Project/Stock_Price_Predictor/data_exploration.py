import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filePath):
    """Loading data from given file path to pandas dataframe""" 
    df =  pd.read_csv(filePath)
    return df

def transform_data(dataset):
    """ Transform data column order in order to separate out features and labels."""
    dataset['Relative_Date'] = dataset.index
    # `Date` to `Low` depicts feature set and 'Close' or 'Adj Close' depicts label set while 'Adj Close' being the prominent one.
    column_order = ['Date','Relative_Date','Volume','Open','High','Low','Close','Adj Close']
    dataset = dataset[column_order]
    return dataset

def display_statistics(dataset):
    """ Display general statistics for data like min, max, mean, standard deviation."""
    stock_data_columns = ['Open','High','Low','Close','Volume']
    print("Data Statistics...\n")
    for column in stock_data_columns:
        print("{} --->".format(column))
        print("Min: ", np.min(dataset[column]), "\t Mean :", np.mean(dataset[column]),"\t Std: ", np.std(dataset[column]),"\t Max: ", np.max(dataset[column]))        

def visualize(dataset, title, column, x_label, y_label):  
    """ Helps in visualizing any column like 'Adj Close' as a time series in a curve."""
    ax = dataset[column].plot(title=title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


