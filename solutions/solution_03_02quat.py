## Even better splitting strategy

from IPython.display import Image
Image('image/BlockedTimeSeriesSplit.png')

# we define our own splitter class
class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]
            
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
y = np.array(features['actual'])

# Remove the labels from the features
# axis 1 refers to the columns
X= features.drop(['Unnamed: 0', 'year', 'month', 'day',
       'actual', 'forecast_noaa', 'forecast_acc', 'forecast_under',
       'week_Fri', 'week_Mon', 'week_Sat', 'week_Sun', 'week_Thurs',
       'week_Tues', 'week_Wed'], axis = 1)


X_train=np.array(X)[:int(len(X.index)*0.75),:]                                                                           
X_test=np.array(X)[int(len(X.index)*0.75):,:]
y_train=np.array(y)[:int(len(X.index)*0.75)]
y_test=np.array(y)[int(len(X.index)*0.75):]
grid_values = {'criterion': ['mse'],
               'n_estimators':np.arange(600,1200,300), 
               'max_depth':np.arange(2,22,5),
               'min_samples_split':np.arange(2,20,4),
              'min_samples_leaf':np.arange(1,20,4)}# define the hyperparameters you want to test
#with the range over which you want it to be tested.
tscv = BlockingTimeSeriesSplit(n_splits=5)


    
#Feed it to the GridSearchCV with the right
#score over which the decision should be taken    
grid_tree_acc = GridSearchCV(RandomForestRegressor(), 
                            param_grid = grid_values, 
                             scoring='r2',
                             cv=tscv, 
                             n_jobs=-1)

grid_tree_acc.fit(X_train, y_train)



print('Grid best parameter (max. r2): ', grid_tree_acc.best_params_)#get the best parameters
print('Grid best score (r2): ', grid_tree_acc.best_score_)#get the best score calculated from the train/validation
#dataset

y_decision_fn_scores_acc=grid_tree_acc.score(X_test,y_test)
print('Grid best parameter (max. r2) model on test: ', y_decision_fn_scores_acc)# get the equivalent score on the test
#dataset : again this is the important metric

## looking at feature importance 
RF = grid_tree_acc.best_estimator_
W=RF.feature_importances_#get the weights

sorted_features=sorted([[list(X.columns)[i],abs(W[i])] for i in range(len(W))],key=itemgetter(1),reverse=True)

print('Features sorted per importance in discriminative process')
print(sorted_features)


from sklearn.inspection import permutation_importance

feature_importance = W
std = np.std([tree.feature_importances_ for tree in grid_tree_acc.best_estimator_.estimators_], axis=0)

sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx],xerr=std[sorted_idx][::-1], align='center')
plt.yticks(pos, np.array(list(X.columns))[sorted_idx])
plt.title('Feature Importance (MDI)',fontsize=10)

result = permutation_importance(RF, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(list(X.columns))[sorted_idx])
plt.title("Permutation Importance (test set)",fontsize=10)
fig.tight_layout()
plt.show()


## plotting the fit
plt.plot(y,RF.predict(X),'ro')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title(str(sc.stats.pearsonr(y,RF.predict(X))[0]))