class Node:
    """A decision tree node."""

    def console_viz(self, rule = ""):
        if self.leaf:
            print(rule, "==>", self.predicted_class_name,"\n")
        else:
            if len(rule) == 0:
                rule = "#" 
            else:
                rule = rule + " && "
            self.left_child.console_viz(rule + self.feature_name + "<{:.2f}".format(self.value) )
            self.right_child.console_viz(rule + self.feature_name + ">={:.2f}".format(self.value))

    def __init__ (self, leaf : bool = False):
        self.leaf = leaf

    def _node(self, gini: float,
          value: float,
          nb_samples: int,
          feature_index: int,
          feature_name: str
        ) :
        self.gini = gini
        self.nb_samples = nb_samples
        self.feature_index = feature_index
        self.feature_name = feature_name
        self.value = value
    
    def _leaf(self, predicted_class: int,
          predicted_class_name: str,
          nb_samples : int
        ):
        self.predicted_class = predicted_class
        self.predicted_class_name = predicted_class_name
        self.nb_samples = nb_samples

    def _children(self, left_child, right_child):
        self.left_child = left_child
        self.right_child = right_child
        self.nb_sample_left = left_child.nb_samples
        self.nb_sample_right = right_child.nb_samples

