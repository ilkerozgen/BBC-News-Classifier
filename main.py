import numpy as np
import pandas as pd

''' 
    Function to get the x and y data.
    Returns them as pandas dataframes. 
'''
def getData(x, y):
    X = pd.read_csv(x, header=0, delimiter=" ")
    y = pd.read_csv(y, header=None, delimiter=" ")

    return X.values, y.values.flatten()

''' 
    Function to train and test a Multinomial Naive Bayes model. 
    Returns the accuracy and the confusion matrix. 
'''
def multinomialNaiveBayes(X_train, y_train, X_test, y_test):
    numberOfClasses = len(np.unique(y_train))
    numberOfWords = X_train.shape[1]
    
    # Calculate the priors
    priors = np.zeros(numberOfClasses)

    for i in range(numberOfClasses):
        priors[i] = np.sum(y_train == i) / len(y_train)
    
    # Calculate the probabilities
    probabilities = np.zeros((numberOfClasses, numberOfWords))

    for i in range(numberOfClasses):
        probabilities[i] = (np.sum(X_train[y_train == i], axis=0)) / (np.sum(X_train[y_train == i]) + numberOfWords)

    # Make the predictions
    logProbabilities = np.dot(X_test, np.log(probabilities).T) + np.log(priors)
    predictions = np.argmax(logProbabilities, axis=1)
    
    # Calculate the accuracy
    accuracy = np.mean(predictions == y_test)
    
    # Calculate the confusion matrix
    confusionMatrix = np.zeros((numberOfClasses, numberOfClasses))

    for i in range(len(y_test)):
        confusionMatrix[y_test[i], predictions[i]] += 1
    
    return accuracy, confusionMatrix

''' 
    Function to extend the Multinomial Naive Bayes model with fair Dirichlet prior.
    Returns the accuracy and the confusion matrix. 
'''
def dirichletPriorNaiveBayes(X_train, y_train, X_test, y_test, alpha=1):
    numberOfClasses = len(np.unique(y_train))
    numberOfWords = X_train.shape[1]
    
    # Calculate the priors
    priors = np.zeros(numberOfClasses)

    for i in range(numberOfClasses):
        priors[i] = (np.sum(y_train == i) + alpha) / (len(y_train) + alpha * numberOfClasses)
    
    # Calculate the probabilities
    probabilities = np.zeros((numberOfClasses, numberOfWords))

    for i in range(numberOfClasses):
        probabilities[i] = (np.sum(X_train[y_train == i], axis=0) + alpha) / (np.sum(X_train[y_train == i]) + alpha * numberOfWords)

    # Make the predictions
    logProbabilities = np.dot(X_test, np.log(probabilities).T) + np.log(priors)
    predictions = np.argmax(logProbabilities, axis=1)
    
    # Calculate the accuracy
    accuracy = np.mean(predictions == y_test)
    
    # Calculate the confusion matrix
    confusionMatrix = np.zeros((numberOfClasses, numberOfClasses))

    for i in range(len(y_test)):
        confusionMatrix[y_test[i], predictions[i]] += 1
    
    return accuracy, confusionMatrix

''' 
    Function to train and test a Bernoulli Naive Bayes model. 
    Returns the accuracy and the confusion matrix. 
'''
def bernoulliNaiveBayes(X_train, y_train, X_test, y_test, alpha=1):
    numberOfClasses = len(np.unique(y_train))
    numberOfWords = X_train.shape[1]
    
    # Calculate the priors
    priors = np.zeros(numberOfClasses)

    for i in range(numberOfClasses):
        priors[i] = (np.sum(y_train == i) + alpha) / (len(y_train) + alpha * numberOfClasses)
    
    # Calculate the probabilities
    probabilities = np.zeros((numberOfClasses, numberOfWords))
    
    for i in range(numberOfClasses):
        probabilities[i] = (np.sum(X_train[y_train == i] > 0, axis=0) + alpha) / (np.sum(y_train == i) + alpha * 2)

    # Make the predictions
    logProbabilities = np.dot(X_test, np.log(probabilities).T) + np.log(priors)
    predictions = np.argmax(logProbabilities, axis=1)
    
    # Calculate the accuracy
    accuracy = np.mean(predictions == y_test)
    
    # Calculate the confusion matrix
    confusionMatrix = np.zeros((numberOfClasses, numberOfClasses))
    
    for i in range(len(y_test)):
        confusionMatrix[y_test[i], predictions[i]] += 1
    
    return accuracy, confusionMatrix

# Get the data
X_train, y_train = getData('dataset/x_train.csv', 'dataset/y_train.csv')
X_test, y_test = getData('dataset/x_test.csv', 'dataset/y_test.csv')

# Train and test Multinomial Naive Bayes model.
accuracy1, confusionMatrix1 = multinomialNaiveBayes(X_train, y_train, X_test, y_test)
print("Multinomial Naive Bayes")
print(f"Accuracy: {accuracy1:.3f}")
print("Confusion Matrix:")
print(confusionMatrix1)

# Train and test Multinomial Naive Bayes model that is extended with fair Dirichlet prior.
accuracy2, confusionMatrix2 = dirichletPriorNaiveBayes(X_train, y_train, X_test, y_test, alpha=1)
print("\nMultinomial Naive Bayes that is extended with Fair Dirichlet Prior")
print(f"Accuracy: {accuracy2:.3f}")
print("Confusion Matrix:")
print(confusionMatrix2)

# Train and test Bernoulli Naive Bayes model.
accuracy3, confusionMatrix3 = bernoulliNaiveBayes(X_train, y_train, X_test, y_test, alpha=1)
print("\nBernoulli Naive Bayes")
print(f"Accuracy: {accuracy3:.3f}")
print("Confusion Matrix:")
print(confusionMatrix3)
