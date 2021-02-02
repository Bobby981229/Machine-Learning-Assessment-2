"""
Title: CMP3751M_Machine Learning_Assessment 2
Task: Section 3
Author: Shangyuan Liu
School: University of Lincoln, School of Computer Science
ID_No: 25344136
E-mail: 25344136@students.lincoln.ac.uk
"""
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
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
    # data_normal['Status'] = data_status  # Add Status list into data_normal

    data_status = data_status.apply(lambda x: 0 if 'healthy' in x else 1)  # Convert healthy = 0, cancerous = 1

    # Shuffle data frame and split the training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(data_normal, data_status, test_size=0.1)
    return x_train, x_test, y_train, y_test


def ann_model(x_train, x_test, y_train, y_test, epochs=200):
    """Artificial Neural Network (ANN) Classifier"""
    acc_array = []  # Record accuracy
    global accuracy
    for epoch in range(1, epochs, 20):
        # Initialising the model
        ann = MLPClassifier(shuffle=True, hidden_layer_sizes=[500, 500], activation='logistic',
                            solver='lbfgs', alpha=0.1, random_state=1, max_iter=epoch)
        ann.fit(x_train, y_train)  # Fit the model to train_data matrix X and target y
        accuracy = ann.score(x_test, y_test)  # Return the average accuracy
        acc_array.append(accuracy)  # Add the accuracy values into list
    print('ANN Model Accuracy: {:.3f}'.format(accuracy))
    # Generate the plot
    plt.plot(range(len(acc_array)), acc_array, color='r', label='Accuracy')  # r
    plt.xlabel('epochs'), plt.ylabel('accuracy'), plt.title("Epochs - Accuracy Plot")
    plt.legend(), plt.show()  # Display plot
    return


def forests_model(x_train, x_test, y_train, y_test):
    """Random Forests Classifier"""
    # Initialising the model
    clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                                 min_samples_split=5, bootstrap=True, oob_score=True)
    scores1 = cross_val_score(clf, x_train, y_train)  # Cross validation
    clf.fit(x_train, y_train)  # Fit the classifier model with training data
    y_pred = clf.predict(x_test)  # Using the trained classifier to predict the labels
    report = classification_report(y_test, y_pred)  # Report the precision ...
    print(report)
    print(clf.oob_score_)  # Report the accuracy
    return


def main():
    """Set the parameters and call the functions"""
    x_train, x_test, y_train, y_test = data_processing()
    ann_model(x_train, x_test, y_train, y_test)
    forests_model(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    """Execute the current module"""
    main()
