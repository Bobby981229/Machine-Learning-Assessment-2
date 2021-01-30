"""
Title: CMP3751M_Machine Learning_Assessment 2
Task: Section 3
Author: Shangyuan Liu
School: University of Lincoln, School of Computer Science
ID_No: 25344136
E-mail: 25344136@students.lincoln.ac.uk
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from CMP3751M_ML_2_Section1 import max_min_normalization, import_data


"""
Section 3
Designing algorithms
"""


def data_processing():
    """Data Normalisation & Shuffle, Split Data"""
    [df, data] = import_data()  # Import the origin data
    data_status = df['Status']  # Status list
    data_normal = max_min_normalization(data)  # Data Normalisation
    data_normal['Status'] = data_status  # Add Status list into data_normal

    # Convert healthy = 0, cancerous = 1
    data_normal['Status'] = data_normal['Status'].apply(lambda x: 0 if 'healthy' in x else 1)

    # Shuffle data frame and split tge training set and testing set
    data_random = data_normal.sample(frac=1).reset_index(drop=True)  # Shuffle the data
    X_train, X_test, y_train, y_test = train_test_split(data_random, data_status, test_size=0.1,
                                                        random_state=0)  # Split data set

    ann_model = MLPClassifier(hidden_layer_sizes=[500, 500], activation='logistic', solver='sgd', alpha=1e-4,
                              random_state=1, max_iter=200, learning_rate_init=0.001)

    ann_model.fit(X_train, y_train)
    print('准确率：{:.3f}'.format(ann_model.score(X_test, y_test)))

    return X_train, X_test, y_train, y_test


def sigmoid(x):
    # sigmoid function as the non-linear activation function#
    return 1 / (1 + np.exp(-x))


def main():
    """Set the parameters and call the functions"""
    data_processing()  # Section 1


if __name__ == '__main__':
    """Execute the current module"""
    main()
