import pandas as pd
import threading
import queue
from DiffPM import DiffPID3 

# This class represents an ensemble structe of private random decision trees.
# It allows the parallel creation and training of the contained trees.
#
# Parameters for constructor:
# number_trees: Number of random decision trees, which should be used
# max_depth: The depth of each decision tree
# attributes: The full list of available attributes from which the decision
#     trees can choose from
# e: the epsilon parameter for the Laplacian distribution
class RandomForest(object):
    def __init__(self, X, Y, feat_names, max_depth, e, number_trees):
        self.trees = list() #list of containing trees
        # the more trees are used, the higher the noise
        self.e_per_tree = e / number_trees
        # Queue for the output of the parallel threads
        que = queue.Queue() 
        for i in range(0, number_trees):
            thr = threading.Thread(target = lambda q, 
                    arg : q.put(DiffPID3(arg)), 
                    args = (que, [X, Y, feat_names, max_depth, self.e_per_tree]))
            thr.start()
            thr.join()
        
        while len(self.trees) < number_trees: 
            # wait until all trees are build and trained 
            self.trees.append(que.get())

    # This function accepts a pandas DataFrame object 
    # with the instances for predictions
    # and returns a list with the estimated classes.
    def predict(self, df):
        predictions = list()
        for (idx, row) in df.iterrows():
            # the variable vote collects the votes of all trees.
            # This can easily adapt to parallel execution.
            vote = 0
            for tree in self.trees:
                vote += tree.predict(row) # prediction for a specific
                                          # instance for each tree
            # Compute prediction
            # If vote is greater than the halt number of trees,
            # more of the half trees voted for class 1. 
            # Otherwise the prediction is 0.
            # It is also possible to use a weighted majority vote
            # or use the posterior probability over all known instances
            if vote > len(self.trees) / 2:
                predictions.append(1)
            else:
                predictions.append(0)
        return pd.Series(predictions, dtype='int64')