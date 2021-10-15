## Our splitting strategy doesn't seem to represent the reality of the process....
## inspired from https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/

## our splitting strategy before : randomly picking points. 
## proposed splitting strategy:
from IPython.display import Image
Image('image/TimeSeriesSplit.png')


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
y = np.array(features['actual'])

# Remove the labels from the features
# axis 1 refers to the columns
X= features.drop(['Unnamed: 0', 'year', 'month', 'day',
       'actual', 'forecast_noaa', 'forecast_acc', 'forecast_under',
       'week_Fri', 'week_Mon', 'week_Sat', 'week_Sun', 'week_Thurs',
       'week_Tues', 'week_Wed'], axis = 1)

## the train data is the 75% most ancient data, the test is the 25% most recent
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
tscv = TimeSeriesSplit()
    
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


## feature importances
RF = grid_tree_acc.best_estimator_
W=RF.feature_importances_#get the weights

sorted_features=sorted([[list(X.columns)[i],abs(W[i])] for i in range(len(W))],key=itemgetter(1),reverse=True)

print('Features sorted per importance in discriminative process')
for f,w in sorted_features:
    print('{:>20}\t{:.3f}'.format(f,w))
    
from sklearn.inspection import permutation_importance

feature_importance = W
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
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

