"""
This module defines the following routines used by the 'train' step of the regression pipeline:

- ``estimator_fn``: Defines the customizable estimator type and parameters that are used
  during training to produce a model pipeline.
"""

def tree_estimator():
  from sklearn.tree import DecisionTreeRegressor
  
  return DecisionTreeRegressor(random_state=42, criterion='mse', max_depth=5)


def sgd_estimator():
    """
    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.
    """
    from sklearn.linear_model import SGDRegressor
    
    return SGDRegressor(random_state=42)

"""
estimator_fn() is the entrypoint for train.py

By defining two estimators, we can demo starting with sgd_estimator, then move to the decision tree which gives better performance for this case
"""
def estimator_fn():
#   return sgd_estimator()
  return tree_estimator()
