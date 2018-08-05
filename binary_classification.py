import numpy as np
from livelossplot import PlotLosses


def image_to_vector(X):
    X = np.reshape(X, (len(X), -1))  # Flatten: (N x 28 x 28) -> (N x 784)
    return np.c_[X, np.ones(len(X))]  # Append 1: (N x 784) -> (N x 785)

data = np.load('../dataset/mnist.npz')
Xtrain = image_to_vector(data['train_x'])
Ytrain = data['train_y']
Xtest = image_to_vector(data['test_x'])
Ytest = data['test_y']

W = np.random.randn(10, 28 * 28 + 1)
Confusion_matrix = np.zeros((10,10), dtype=np.int32)

eta = 0.001 # hyperparameter
# liveloss = PlotLosses()
for t in range(100):
    # Structured perceptron for updating weights.
    num_correct_train = 0
    for x, y in zip(Xtrain, Ytrain):
        y_pred = np.argmax(np.dot(W, x)) # np.argmax return the index max element
        if y_pred != y:
            W[y] += x * eta
            W[y_pred] -= x * eta
        else:
            num_correct_train += 1

    # Evaluate and store the accuracy on the test set.
    num_correct_test = 0
    for x, y in zip(Xtest, Ytest):
        y_pred = np.argmax(np.dot(W, x))
        if y_pred == y:
            num_correct_test += 1
        Confusion_matrix[y_pred][y] += 1

    # Visualize accuracy values on the training and test sets.
    # liveloss.update({
    #     'accuracy': float(num_correct_train) / len(Ytrain),
    #     'val_accuracy': float(num_correct_test) / len(Ytest)
    # })
    # liveloss.draw()

print('Accuracy: {:.4f} (test), {:.4f} (train)'.format(
    float(num_correct_test) / len(Ytest),
    float(num_correct_train) / len(Ytrain)
))

print(Confusion_matrix)

# result: test  accuracy: 0.8883
#         train accuracy: 0.9002
# variability of the result is due to random initialization of W
