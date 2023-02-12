'''
https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html
Application: Exploring Hand-written Digits
'''


from sklearn.datasets import load_digits
digits = load_digits()

import matplotlib.pyplot as plt

fig, axes = plt.subplots(10, 10, figsize=(8, 8),  subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]), transform=ax.transAxes, color='green')

plt.show()


X = digits.data
y = digits.target


from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)



plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,
            edgecolor='none', alpha=0.5,
            #cmap=plt.cm.get_cmap('spectral', 10))
            cmap = plt.cm.get_cmap('jet', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5);


plt.show()

from sklearn.model_selection  import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)


from sklearn.metrics import accuracy_score
print("accuracy_score: "+ str(accuracy_score(ytest, y_model)))




from sklearn.metrics import confusion_matrix

mat = confusion_matrix(ytest, y_model)

import seaborn as sns
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value');

plt.show()








fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

test_images = Xtest.reshape(-1, 8, 8)

for i, ax in enumerate(axes.flat):
    ax.imshow(test_images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(y_model[i]),
            transform=ax.transAxes,
            color='green' if (ytest[i] == y_model[i]) else 'red')


plt.show()






