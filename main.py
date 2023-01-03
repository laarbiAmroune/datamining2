import pandas as pd
from src.modal import *

# Load data.
train_size = 100
df = pd.read_csv("diabetes.csv", delimiter=",")
features = list(df.keys())[:-1]
data = df.to_numpy()
X = data[:, :-1] # all columns but the last
y = data[:, -1].astype(int) # expected to be from 0 to n_classes - 1
# Fit data.
print(data[:1, :-1]**2)
print(X.shape, y.shape)
clf = DTree(max_depth = 10)
clf.meta_data(feature_names = features, class_names = ['no diabetes', 'diabetes'])
clf.fit(X, y)
#print(clf.predict(data[10:30,:-1]))
#print(list(y[10:30]))
print(np.sum((y-clf.predict(data[:,:-1]))**2)/len(y))
'''
print('x shape= ', X.shape)
clf = DTree(max_depth=2)
clf.fit(X, y)

# Visualize.
clf.debug(
    feature_names=features,
    class_names=["d 0", "d 1"],
)
#'''
'''
print("-----------------------------------------------------")
print(clf.predict([X[3]]))
print(clf.predict([X[3]]))
print(clf.predict([X[0]]))
print(clf.predict([X[20]]))
'''