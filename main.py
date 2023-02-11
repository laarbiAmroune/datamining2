import pandas as pd
from src.modal import *

# Load data.
depth = 8

train_percentage = 0.8
df = pd.read_csv("diabetes.csv", delimiter=",")
features = list(df.keys())[:-1]
data = df.to_numpy()

X_train = data[:int(data.shape[0]*train_percentage), :-1]
y_train= data[:int(data.shape[0]*train_percentage), -1].astype(int)

X_test = data[int(data.shape[0]*train_percentage) :, :-1]
y_test = data[int(data.shape[0]*train_percentage) :, -1].astype(int)

# Fit data.
clf = Dicision_tree(max_depth = depth)
clf.meta_data(feature_names = features, class_names = ['no diabetes', 'diabetes'])
clf.fit(X_train, y_train)

print("prediction dans l'ensemble de test\n",clf.predict(X_test))
print("erreur sur l'ensemble d'entrainement:", np.sum((y_train-clf.predict(X_train))**2)/len(y_train) )
print("erreur quadratique de l'ensemble de test:",np.sum((y_test-clf.predict(X_test))**2)/len(y_test))

print("\n\n\n")
print("affichage de l'arbre sous forme de regles")
clf.console_viz()