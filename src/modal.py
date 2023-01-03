import numpy as np
import pickle
from src.node import *

class DTree :
    def __init__(self, max_depth=None, load_path = None):
        #if given file loading path
        if load_path != None :
            self._root = self._load(load_path)
        elif max_depth != None:
            self.max_depth = max_depth
    
    #load modal from file
    def _load(self, load_path: str):
        with open(load_path, 'rb') as modal:
            return pickle.load(modal)

    #save modal into file
    def _save(self, save_path: str):
         with open(load_path, 'wb') as modal:
            pickle.dump(self._root, modal)
    
    def debug(self, feature_names, class_names, show_details=True):
        """#print ASCII visualization of decision tree."""
        self._root.debug(feature_names, class_names, show_details)

    def _optimal_split(self, X, y):
        # Need at least two elements to split a node.
        nb_samples = y.size
        if nb_samples <= 1:
            return None, None
        
        # Count of each class in the current node.
        num_parent = [np.sum(y == c) for c in range(self._nb_classes)]

        # Gini of current node.
        best_gini = 1.0 - sum((n / nb_sample) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self._nb_features):
            # Sort data along selected feature.
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            # We could actually split the node according to each feature/threshold pair
            # and count the resulting population for each class in the children, but
            # instead we compute them in an iterative fashion, making this for loop
            # linear rather than quadratic.
            num_left = [0] * self._nb_classes
            num_right = num_parent.copy()
            for i in range(1, nb_sample):  # possible split positions
                c = int(classes[i - 1])
                num_left[c] += 1
                num_right[c] -= 1
                ##print(idx, i, c,num_left[c],num_right[c])
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self._nb_classes)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (nb_samples - i)) ** 2 for x in range(self._nb_classes)
                )

                # The Gini impurity of a split is the weighted average of the Gini
                # impurity of the children.
                gini = (i * gini_left + (nb_samples - i) * gini_right) / nb_sample

                # The following condition is to make sure we don't try to split two
                # points with identical values for that feature, as it is impossible
                # (both have to end up on the same side of a split).
                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

        return best_idx, best_thr

    def optimal_split(self, features, labels):
        optimal_gini = self._gini(labels)#the gini value of the current node
        optimal_split_feature = None
        optimal_split_value = None
        optimal_split_size = None
        count_classes = [np.sum(labels == c) for c in set(labels)]

        nb_samples = len(labels)

        for feature_ in range(self._nb_features):
            print(self.feature_names[feature_])
            #maching feature_ and labels
            ##print("features", features)
            ##print("labels", labels)
            zipped = zip(features[:, feature_],labels)

            #sorting on the basis of the feature value
            arr = sorted(zipped)
            f_s,l_s = zip(*arr)
            ##print(f_s, l_s)
            ##print(len(f_s) == len(features[feature_]))

            count_left = np.zeros(self._nb_classes)
            #sample index
            s_i = 0
            while s_i < nb_samples :
                ##print(s_i,len(f_s), len(l_s), s_i < len(f_s))
                """ when spliting all element of the same value
                for the feature should be grouped together,
                so navigating to the last element of the same value"""
                while s_i < nb_samples - 1 and f_s[s_i] == f_s[s_i + 1]:
                    count_left[l_s[s_i]] += 1
                    s_i +=1

                #evaluation the gini of the children
                count_left[l_s[s_i]] += 1
                #print("#",count_classes)
                #print("*",count_left*count_left, "/", s_i, "|", len(labels))
                left_potential_gini = 1.0 - np.sum(count_left**2) /((s_i + 1)**2)
                right_potential_gini = 1.0 - np.sum((count_classes - count_left) ** 2)/((nb_samples - s_i) ** 2)
                potential_gini = (
                        (s_i) * left_potential_gini 
                        +(nb_samples - s_i) * right_potential_gini
                    )/nb_samples
                if potential_gini < optimal_gini:
                    optimal_gini = potential_gini
                    optimal_split_feature = feature_
                    optimal_split_size = s_i + 1
                    optimal_split_value = (f_s[s_i] + f_s[s_i + 1])/2
                """print(optimal_gini,
                    optimal_split_size,
                    optimal_split_feature)"""
                s_i += 1
        return {
            'gini': optimal_gini,
            'feature': optimal_split_feature,
            'size': optimal_split_size,
            'value': optimal_split_value
        }

    def _gini(self, y):
        nb_samples = y.size
        
        # Count of each class in the current node.
        num_parent = [np.sum(y == c) for c in range(self._nb_classes)]

        return  1.0 - sum((n / nb_samples) ** 2 for n in num_parent)



    def _fit(self, X, y):
        self._nb_classes = len(set(y))
        self._nb_features = X.shape[1]
        self._root = self._grow_tree(X, y)

    def fit(self, features, labels):
        self._nb_classes =len(set(labels))
        self._nb_features = features.shape[1]
        self._root = self.construct_tree(features, labels)

    def _grow_tree(self, X, y, depth=0):
        """Build a decision tree by recursively finding the best split."""
        # Population for each class in current node. The predicted class is the one with
        # largest population.
        num_samples_per_class = [np.sum(y == i) for i in range(self._nb_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        # Split recursively until maximum depth is reached.
        if depth < self.max_depth:
            idx, thr = self._optimal_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    
    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        """Predict class for a single sample."""
        print("prediction", inputs)
        node = self._root
        while not node.leaf:
            #print(node.leaf)
            print(node.feature_name," = ", node.value)
            if inputs[node.feature_index] < node.value:
                node = node.left_child
            else:
                node = node.right_child
        #print(node.predicted_class_name)
        return node.predicted_class


    def meta_data(self, feature_names: list = [], class_names: list = []):
        self.feature_names = feature_names
        self.class_names = class_names
    
    def construct_tree(self, features, labels, depth = 0):
        #count the number of different inctences of classes in labels
        count_classes = np.array([np.sum(labels == c) for c in set(labels)])

        #case we reached the max depth
        if depth >= self.max_depth :
            _class = count_classes.argmax()
            node = Node(leaf = True)
            node._leaf(predicted_class = _class, predicted_class_name = self.class_names[_class], nb_samples = len(labels))
            return node
        #case only one class in labels
        if count_classes.max() == len(labels):
            _class = count_classes.argmax()
            node = Node(leaf = True)
            node._leaf(predicted_class = _class, predicted_class_name = self.class_names[_class], nb_samples = len(labels))
            return node

        #case possible split
        
        split = self.optimal_split(features, labels)
        if split['feature'] != None:
            node = Node()
            node._node(gini = split['gini'],
                    value = split['value'],
                    nb_samples = len(labels),
                    feature_index = split['feature'],
                    feature_name = self.feature_names[split['feature']]
                )
            #sorting feature
            sort_arg = features[:, split['feature']].argsort()
            sorted_features = features.copy()[sort_arg]
            sorted_labels = np.transpose(labels.copy()[sort_arg])
            #print(labels, sorted_labels)

            #spliting for left and right
            features_left = sorted_features[:split['size'],:]
            labels_left = sorted_labels[:split['size']]
            features_right = sorted_features[split['size']:,:]
            labels_right = sorted_labels[split['size']:]
            #print(features_left.shape, features_right.shape, split['size'])
            print(split)
            node._children(left_child = self.construct_tree(features_left, labels_left, depth = depth + 1),
                right_child = self.construct_tree(features_right, labels_right, depth = depth + 1)
            )

            return node
        else:
            _class = count_classes.argmax()
            node = Node(leaf = True)
            node._leaf(predicted_class = _class, predicted_class_name = self.class_names[_class], nb_samples = len(labels))
            return node
        
'''     
    def predict(self, samples):
        pass
    
    def predict_sample(self, sample):
        pass

    def gini(self, data):
        pass
'''
    