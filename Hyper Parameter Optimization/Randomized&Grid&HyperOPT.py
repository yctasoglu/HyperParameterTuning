from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,RandomizedSearchCV
from xgboost import XGBClassifier
import numpy as np
from hyperopt import hp,tpe,Trials,fmin
from sklearn.datasets import make_classification
import time
X,Y=make_classification(n_samples=300,
                        n_features=25,
                        n_informative=2,
                        n_redundant=10,
                        n_classes=2,
                        random_state=8)

classifier = XGBClassifier()
metric = 'neg_log_loss'

#Parameter search space for both GridSearch and RandomizedSearch
grid_values = {'objective': ['binary:logistic'],
               'colsample_bytree': [0.6, 1.0], 
               'gamma': [0, 1], 
               'learning_rate':  [0.01, 0.1, 0.3],
               'max_depth': [4,  6], 'min_child_weight':[1, 5], 
               'n_estimators': [90,120,150],
               'random_state': [1],
               'subsample': [1.0]}

max_evals = 100

random_search = RandomizedSearchCV(classifier,param_distributions = grid_values,n_iter = max_evals,scoring = metric)
start = time.time()
random_search.fit(X,Y)
elapsed_time_random = time.time() - start

grid_search = GridSearchCV(classifier,param_grid = grid_values,scoring = metric)
start = time.time()
grid_search.fit(X,Y)
elapsed_time_grid = time.time() - start


#xgboost search space for Hyperopt
xgboost_space={
            'max_depth': hp.choice('x_max_depth',[2,3,4,5,6]),
            'min_child_weight':hp.choice('x_min_child_weight',np.round(np.arange(0.0,0.2,0.01),5)),
            'learning_rate':hp.choice('x_learning_rate',np.round(np.arange(0.005,0.3,0.01),5)),
            'subsample':hp.choice('x_subsample',np.round(np.arange(0.1,1.0,0.05),5)),
            'colsample_bylevel':hp.choice('x_colsample_bylevel',np.round(np.arange(0.1,1.0,0.05),5)),
            'colsample_bytree':hp.choice('x_colsample_bytree',np.round(np.arange(0.1,1.0,0.05),5)),
            'n_estimators':hp.choice('x_n_estimators',np.arange(25,100,5))
            }

best_score = 1.0

def objective(space):
    global best_score
    model = XGBClassifier(**space, n_jobs=-1)
    kfold = KFold(n_splits=3, random_state=1985, shuffle=True)
    score = -cross_val_score(model, X, Y, cv=kfold, scoring='neg_log_loss', verbose=False).mean()
    if (score < best_score):
        best_score = score
    return score

start = time.time()
best_params_hyperopt = fmin(
  objective,
  space = xgboost_space,
  algo = tpe.suggest,
  max_evals = max_evals,
  trials = Trials())
elapsed_time_hyperopt = time.time() - start


print("Grid")
print(grid_search.best_score_, grid_search.best_params_)
print("GridSearchCV took %.2f seconds" % (elapsed_time_grid))

print("Random")
print(random_search.best_score_, random_search.best_params_)
print("RandomizedSearchCV took %.2f seconds for %d candidates" % (elapsed_time_random, max_evals))

print("HyperOpt")
print(-best_score,best_params_hyperopt)
print("HyperOPT took %.2f seconds for %d candidates" % (elapsed_time_hyperopt, max_evals))
