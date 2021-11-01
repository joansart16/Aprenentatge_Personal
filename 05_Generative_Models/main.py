import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(8, 8),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)


digits = datasets.load_digits()
plot_digits(digits.data[:100, :])
pca = PCA()

#Preprocessament
digits2 = []
for d in digits.images:
    digits2.append(np.reshape(d, 64))

#print (digits.keys())

X_pca = pca.fit_transform(digits.data)
n_row, n_col = 8, 8
n_components = n_row * n_col
colors = ['black', 'blue', 'purple', 'yellow', 'green', 'red', 'lime', 'cyan', 'orange', 'gray']
plt.figure()
for i in range(len(colors)):
    px = X_pca[:, 0][digits.target == i]
    py = X_pca[:, 1][digits.target == i]
    plt.scatter(px, py, c=colors[i])
plt.legend(digits.target_names)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

n_row, n_col = 8, 8
n_components = n_row * n_col
plt.figure(figsize=(2. * n_col, 2.26 * n_row))
for i, comp in enumerate(pca.components_[:pca.n_components_]):
    plt.subplot(n_row, n_col, i + 1)
    plt.imshow(comp.reshape((8, 8)), interpolation='nearest')
    plt.text(0, -1, str(i + 1) + '-component')
    plt.xticks(())
    plt.yticks(())



plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
# print(np.cumsum(pca.explained_variance_ratio_))

print(X_pca.shape)
pca = PCA(0.99, whiten=True)
data = pca.fit_transform(digits.data)

print(data.shape)



#GaussianMixture
# n_components = np.arange(1, 500, 50)
# models = [GaussianMixture(n, covariance_type='full', random_state=0)
#           for n in n_components]
# aics = [model.fit(data).bic(data) for model in models]
# plt.figure()
# plt.plot(n_components, aics)
#
# n_components = np.arange(1, 20, 1)
# models = [GaussianMixture(n, covariance_type='full', random_state=0)
#           for n in n_components]
# aics = [model.fit(data).bic(data) for model in models]
# plt.figure()
# plt.plot(n_components, aics)
#
# n_components = np.arange(65, 75, 1)
# models = [GaussianMixture(n, covariance_type='full', random_state=0)
#           for n in n_components]
# aics = [model.fit(data).bic(data) for model in models]
# plt.figure()
# plt.plot(n_components, aics)

gmm = GaussianMixture(72, covariance_type='full', random_state=0)
gmm.fit(data)
print(gmm.converged_)

data_new, _ = gmm.sample(100)
print(data_new.shape)

digits_new = pca.inverse_transform(data_new)
plot_digits(digits_new)

plt.show()
