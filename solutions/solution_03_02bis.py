from sklearn.ensemble import RandomForestRegressor

y = np.array(features['actual'])

# Remove the labels from the features
# axis 1 refers to the columns
X= features.drop(['Unnamed: 0', 'year', 'month', 'day',
       'actual', 'forecast_noaa', 'forecast_acc', 'forecast_under',
       'week_Fri', 'week_Mon', 'week_Sat', 'week_Sun', 'week_Thurs',
       'week_Tues', 'week_Wed'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                                           random_state = 42)
grid_values = {'criterion': ['squared_error'],
               'n_estimators':[300,600,900], 
               'max_depth':[2,5,7],
               'min_samples_split':[4],
              'min_samples_leaf':[2]}# define the hyperparameters you want to test
#with the range over which you want it to be tested.

grid_tree_acc = GridSearchCV(RandomForestRegressor(), param_grid = grid_values, scoring='r2',n_jobs=-1)#Feed it to the GridSearchCV with the right
#score over which the decision should be taken

grid_tree_acc.fit(X_train, y_train)

y_decision_fn_scores_acc=grid_tree_acc.score(X_test,y_test)

print('Grid best parameter (max. r2): ', grid_tree_acc.best_params_)#get the best parameters
print('Grid best score (r2): ', grid_tree_acc.best_score_)#get the best score calculated from the train/validation
#dataset
print('Grid best parameter (max. r2) model on test: ', y_decision_fn_scores_acc)# get the equivalent score on the test
#dataset : again this is the important metric

RF = grid_tree_acc.best_estimator_


W=RF.feature_importances_#get the weights

sorted_features=sorted([[list(X.columns)[i],abs(W[i])] for i in range(len(W))],key=itemgetter(1),reverse=True)

print('Features sorted per importance in discriminative process')
for f,w in sorted_features:
    print('{:>20}\t{:.3f}'.format(f,w))
    
from sklearn.inspection import permutation_importance
feature_importance = RF.feature_importances_
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