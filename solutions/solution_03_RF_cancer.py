from sklearn.ensemble import RandomForestClassifier

grid_values = {'n_estimators' : [10,50,100,150,200], 
               'criterion': ['entropy','gini'],
               'max_depth':np.arange(2,10), ## I reduce the search space in the interest of time too
               'min_samples_split':np.arange(2,12,2)}

grid_tree = GridSearchCV(RandomForestClassifier(class_weight='balanced'), 
                                param_grid = grid_values, 
                                scoring='roc_auc',
                                cv = 5,
                                n_jobs=-1)
grid_tree.fit(X_train_cancer, y_train_cancer)

print(f'Grid best score (accuracy): {grid_tree.best_score_:.3f}')
print('Grid best parameter :')

for k,v in grid_tree.best_params_.items():
    print('{:>25}\t{}'.format(k,v))