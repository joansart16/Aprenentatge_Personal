import numpy as np

class Perceptron:
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.

    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        # TODO: Put your code

        for _ in self.n_iter:

            for n in X:

                z = self.__net_input(self, n)
                if z>=0:
                    y_nova = 1
                else:
                    y_nova = -1
                #Actualizar pesos

                error = y[n]-y_nova
                dist = error*self.eta
                for i_w in self.w_:
                    self.w_[n] = X[i_w]*dist
        return self

    def __net_input(self, X):
        """Calculate net input"""

        # TODO: Put your code
        return np.dot(X,self.w_)
    def predict(self, X):
        """Return class label"""

        # TODO: Put your code
