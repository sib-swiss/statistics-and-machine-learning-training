## linear model
pipeline_SVR=Pipeline([('scalar',StandardScaler()),('model',svm.SVR())])

grid_values = {"model__kernel": ['linear'],
                 "model__C":np.logspace(-2, 2, 20)}

grid_linear_SVR_diabetes_r2 = GridSearchCV(pipeline_SVR, param_grid = grid_values, scoring=sco,n_jobs=-1)

grid_linear_SVR_diabetes_r2.fit(X_diabetes_train, y_diabetes_train)

y_diabeties_decision_fn_scores_linear_SVR_r2=grid_linear_SVR_diabetes_r2.score(X_diabetes_test,y_diabetes_test)
print('Grid best parameter (max.'+sco+'): ', grid_linear_SVR_diabetes_r2.best_params_)
print('Grid best score ('+sco+'): ', grid_linear_SVR_diabetes_r2.best_score_)
print('Grid best parameter (max.'+sco+') model on test: ', y_diabeties_decision_fn_scores_linear_SVR_r2)

## features importance
w_diabetes_linear_SVR=grid_linear_SVR_diabetes_r2.best_estimator_[1].coef_[0]

sorted_features=sorted([[df_diabetes.columns[i],abs(w_diabetes_linear_SVR[i])] for i in range(len(w_diabetes_linear_SVR))],key=itemgetter(1),reverse=True)

print('Features sorted per importance in discriminative process')
for f,w in sorted_features:
    print('{:>20}\t{:.3f}'.format(f,w))