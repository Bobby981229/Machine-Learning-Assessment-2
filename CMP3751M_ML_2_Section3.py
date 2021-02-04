"""
Title: CMP3751M_Machine Learning_Assessment 2
Task: Section 3
Author: Shangyuan Liu
School: University of Lincoln, School of Computer Science
ID_No: 25344136
E-mail: 25344136@students.lincoln.ac.uk
"""

"""
Section 3
Designing algorithms
"""

from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from CMP3751M_ML_2_Section1 import max_min_normalization, import_data
from sklearn import metrics


def data_processing():
    """Data Normalisation & Shuffle, Split Data"""
    [df, data] = import_data()  # Import the origin data
    data_status = df['Status']  # Status list
    data_normal = max_min_normalization(data)  # Data Normalisation
    # data_normal['Status'] = data_status  # Add Status list into data_normal
    # Convert healthy = 0, cancerous = 1
    data_status = data_status.apply(lambda x: 0 if 'healthy' in x else 1)
    # Shuffle data frame and split the training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(data_normal, data_status, test_size=0.1)
    return x_train, x_test, y_train, y_test


def ann_model(x_train, x_test, y_train, y_test, epochs=200):
    """Artificial Neural Network (ANN) Classifier"""
    global accuracy
    acc_array = []
    for epoch in range(1, epochs, 20):
        # Initialising the model
        ann = MLPClassifier(shuffle=True, hidden_layer_sizes=[500, 500], activation='logistic',
                            solver='lbfgs', alpha=0.1, random_state=1, max_iter=epoch)
        ann.fit(x_train, y_train)  # Fit the model to train_data matrix X and target y
        accuracy = ann.score(x_test, y_test)  # Return the average accuracy
        acc_array.append(accuracy)  # Add the accuracy values into list
    print('ANN Model Accuracy: %.2f%%' % (accuracy * 100))
    plt.plot(range(1, epochs, 20), acc_array, color='r', label='Accuracy')  # Generate the plot
    plt.xlabel('epochs'), plt.ylabel('accuracy'), plt.title("Epochs - Accuracy Plot")
    plt.legend(), plt.show()  # Display plot
    return


def forests_model(x_train, x_test, y_train, y_test, tree_num=1000, min_sam=5):
    """Random Forests Classifier"""
    # Initialising the model
    clf = RandomForestClassifier(n_estimators=tree_num, max_depth=None,
                                 min_samples_split=min_sam, bootstrap=True, oob_score=True)
    scores1 = cross_val_score(clf, x_train, y_train)  # Cross validation
    clf.fit(x_train, y_train)  # Fit the classifier model with training data
    y_pred = clf.predict(x_test)  # Using the trained classifier to predict the labels
    acc_test = "%.2f%%" % ((metrics.accuracy_score(y_test, y_pred)) * 100)  # Report the accuracy
    # print('Forests Model Accuracy:', acc_test)
    return acc_test


def forests_plot(x_train, x_test, y_train, y_test, tree_number, min_samples):
    """Create a plot and show performance changes as more trees are added"""
    acc_list_5 = []  # When min_samples_split value is 5
    acc_list_50 = []  # When min_samples_split value is 50
    # Foreach to calculation accuracy
    for i in tree_number:
        for j in min_samples:
            acc_values = forests_model(x_train, x_test, y_train, y_test, i, j)
            if j == 5:
                acc_list_5.append(acc_values)  # Store acc_values when min_samples is 5
            elif j == 50:
                acc_list_50.append(acc_values)  # Store acc_values when min_samples is 50
    # Generate the plot
    plt.plot(tree_number, acc_list_5, color='red', label="min_samples_split: 5", linestyle='--')
    plt.plot(tree_number, acc_list_50, color='blue', label="min_samples_split: 50", linestyle='-.')
    plt.xlabel('Trees'), plt.ylabel('accuracy'), plt.title("Trees - Accuracy Plot")
    plt.legend()
    plt.show()
    return


def main():
    """Set the parameters and call the functions"""
    tree_number = [100, 1000, 2000, 3000, 4000, 5000]
    min_samples = [5, 50]
    x_train, x_test, y_train, y_test = data_processing()  # Data Pre-processing
    # ann_model(x_train, x_test, y_train, y_test)  # ANN Classifier
    acc_sam1 = forests_model(x_train, x_test, y_train, y_test, 1000, 50)
    acc_sam2 = forests_model(x_train, x_test, y_train, y_test, 1000, 5)
    print('Min Samples:50  Forests Model Accuracy:', acc_sam1)
    print('Min Samples:05  Forests Model Accuracy:', acc_sam2)
    forests_plot(x_train, x_test, y_train, y_test, tree_number, min_samples)


if __name__ == '__main__':
    """Execute the current module"""
    main()
