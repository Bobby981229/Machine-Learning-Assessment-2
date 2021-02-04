"""
Title: CMP3751M_Machine Learning_Assessment 2
Task: Section 4
Author: Shangyuan Liu
School: University of Lincoln, School of Computer Science
ID_No: 25344136
E-mail: 25344136@students.lincoln.ac.uk
"""

"""
Section 4
Model selection
"""

from warnings import filterwarnings
from sklearn.ensemble import RandomForestClassifier
filterwarnings('ignore')
from sklearn.neural_network import MLPClassifier
from CMP3751M_ML_2_Section1 import import_data, max_min_normalization
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt


def data_processing():
    """Import data and Pre-Processing"""
    [df, data] = import_data()  # Call function import_data to import data
    data_status = df['Status']  # Status list
    # Convert healthy = 0, cancerous = 1
    data_status = data_status.apply(lambda x: 0 if 'healthy' in x else 1)
    data_normal = max_min_normalization(data)  # Data Normalisation
    return data_normal, data_status


def ann_cross_validation(data_normal, data_status, neurons, folds):
    """Artificial Neural Network (ANN) Classifier"""
    cv_scores = []  # Store the resulting values for each model
    k_folds = KFold(shuffle=True, n_splits=folds, random_state=1)
    for neuron in neurons:  # Apply different number of neuron parameter in ANN model
        ann = MLPClassifier(shuffle=True, hidden_layer_sizes=[neuron, neuron], activation='logistic',
                            solver='lbfgs', alpha=0.1, random_state=0)  # Initialising the model
        score = cross_val_score(ann, data_normal, data_status, cv=k_folds)  # Cross validation
        cv_scores.append(score.mean())
        print("ANN: Neurons:%d Accuracy:%0.2f" % (neuron, score.mean()))  # Report accuracy
    plt.plot(neurons, cv_scores, color='black')  # Plot the accuracy change tend
    plt.xlabel('Neurons')
    plt.ylabel('Accuracy')
    plt.title('ANN Evaluation Plot')
    plt.show()  # Choose the best parameters from the image
    return


def random_forests_cross_validation(data_normal, data_status, trees, folds, min_samples):
    """Random Forests Classifier"""
    cv_scores = []  # Store the resulting values for each model
    k_folds = KFold(shuffle=True, n_splits=folds, random_state=1)
    for tree in trees:  # Apply different number of trees parameter in random forests model
        forest_model = RandomForestClassifier(n_estimators=tree, min_samples_split=min_samples,
                                              bootstrap=True, oob_score=True)  # Initialising the model
        score = cross_val_score(forest_model, data_normal, data_status, cv=k_folds)  # Cross validation
        cv_scores.append(score.mean())
        print("Random Forests: Trees:%d Accuracy:%0.2f" % (tree, score.mean()))  # Report accuracy
    plt.plot(trees, cv_scores, color='r')  # Plot the accuracy change tend
    plt.xlabel('Trees')
    plt.ylabel('Accuracy')
    plt.title('Random Forests Evaluation Plot')
    plt.show()  # Choose the best parameters from the image
    return


def main():
    """Set the parameters and call the functions"""
    data_normal, data_status = data_processing()  # Data Pre-processing
    hidden_neurons = [50, 500, 1000]  # ANN two hidden layers, [50, 500, 1000] neurons.
    trees_number = [20, 500, 10000]  # Random forest [20, 500, 10,000] trees
    # Set parameters for ANN and Random Forests classifier model to evaluate
    ann_cross_validation(data_normal, data_status, hidden_neurons, folds=10)  # Evaluate ANN
    random_forests_cross_validation(data_normal, data_status, trees_number, folds=10, min_samples=5)
    return


if __name__ == '__main__':
    """Execute the current module"""
    main()
