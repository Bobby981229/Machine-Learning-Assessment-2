"""
Title: CMP3751M_Machine Learning_Assessment 2
Task: Section 1
Author: Shangyuan Liu
School: University of Lincoln, School of Computer Science
ID_No: 25344136
E-mail: 25344136@students.lincoln.ac.uk
"""

"""
Section 1
Data import, summary, pre-processing and visualisation
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def import_data():
    # Import the dataset by read_excel function
    df = pd.read_excel('clinical_dataset.xlsx', engine='openpyxl')
    # print("The original dataset: \n{0}".format(df))  # Formatted output
    data = df.drop(['Status'], axis=1)  # Drop the Status list
    return df, data


def data_pre_processing():
    """Import the data from original clinical_dataset.xlsx file"""
    # Pass the loaded data_frame and data
    [df, data] = import_data()

    # Summary of Statistics
    statistics(df['Age'], 'Age')
    statistics(df['BMI'], 'BMI')
    statistics(df['Glucose'], 'Glucose')
    statistics(df['Insulin'], 'Insulin')
    statistics(df['HOMA'], 'HOMA')
    statistics(df['Leptin'], 'Leptin')
    statistics(df['Adiponectin'], 'Adiponectin')
    statistics(df['Resistin'], 'Resistin')
    statistics(df['MCP.1'], 'MCP.1')

    # Report the size and features
    row_data = np.shape(df)[0]  # Rows
    col_data = np.shape(df)[1]  # Columns
    print("\nThe size of the dataset: %d × %d\nThe number of features: %d\n"
          % (row_data, col_data - 1, col_data - 1))

    # Find missing values
    find_missing_value(df)

    # Find categorical variables in the dataset
    list_categories = list(df['Status'].value_counts().index)
    print("\nThere are categorical variables in the Status feature:", list_categories)
    status_dummy = pd.get_dummies(df['Status'], drop_first=False, prefix='Status')
    print(status_dummy)

    # Normalise the data before starting training/testing any model
    print("\nZ-Score Normalization Method:\n", z_score(data))  # By Z-Score method
    print("\nMin-Max Normalization Method:\n", max_min_normalization(data))  # By Min-Max Normalization method

    # Plot the box and density plots
    box_density_plots(df)
    return


def statistics(dataset, name):
    """Calculate a summary of the dataset"""
    mean_value = ("%.2f" % np.mean(dataset))  # Calculate the mean value of array
    std_value = ("%.2f" % np.std(dataset))  # Calculate the standard deviation value of array
    min_value = ("%.2f" % np.min(dataset))  # Calculate the minimum value of array
    max_value = ("%.2f" % np.max(dataset))  # Calculate the maximum value of array
    # Display the summary of statistics
    print("\n———————— The summary statistics of the feature %s ————————" % name)
    print("The mean value of %s is: %s" % (name, mean_value))
    print("The standard deviation value of %s is: %s" % (name, std_value))
    print("The minimum value of %s is: %s" % (name, min_value))
    print("The maximum value of %s is: %s" % (name, max_value))
    return


def find_missing_value(dataset):
    """Find missing values"""
    missing_value = dataset.isnull().any()  # Determining "columns"
    print("The feedback of missing values\n", missing_value)
    # for i in range(np.shape(missing_value)[0]):
    #     if missing_value[i] == 'True':
    #         print('The column %d has missing values' % i)
    #     else:
    #         print('There are no missing values in column %d' % i)
    return


def z_score(dataset):
    """Z-score Normalization"""
    # The standardised data is normally distributed with mean 0 and variance 1
    avg = np.mean(dataset)  # Calculate the mean value
    std = np.std(dataset)  # Calculate the standard deviation value
    data_z_score = (dataset - avg) / std  # Z-Score
    return data_z_score


def max_min_normalization(dataset):
    """[0,1] Normalization"""
    data_normal = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
    return data_normal


def box_density_plots(df):
    """Plot the box and density plots"""
    # Plot the box plot between Status and Age
    df.boxplot('Age', 'Status', notch=False, grid=True, meanline=True, showmeans=True,
               boxprops={'color': 'black', 'linewidth': '2.0'},
               capprops={'color': 'red', 'linewidth': '2.0'},
               flierprops={'marker': '*', 'markerfacecolor': 'red', 'color': '654EA3'},
               meanprops={'marker': 'o', 'markerfacecolor': 'blue'},  # Set mean value point
               medianprops={'marker': 'x', 'linestyle': '--', 'color': '#FF6D70'})  # Set median line
    plt.xlabel("Status", fontsize=12)
    plt.ylabel('Age', fontsize=12)
    plt.show()

    # Plot the density plot of BMI
    bmi_healthy = df[df['Status'] == 'healthy']['BMI']  # BMI data when Status is healthy
    bmi_cancerous = df[df['Status'] == 'cancerous']['BMI']  # BMI data when Status is cancerous
    sns.kdeplot(bmi_healthy, label="Healthy status", color="green", alpha=.7)  # Plot health
    sns.kdeplot(bmi_cancerous, label="Cancerous status", color="red", alpha=.7)  # Plot cancerous
    plt.title('Density Plot of BMI', fontsize=18)
    plt.xlabel('BMI', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.show()
    return


def main():
    """Set the parameters and call the functions"""
    data_pre_processing()  # Section 1


if __name__ == '__main__':
    """Execute the current module"""
    main()
