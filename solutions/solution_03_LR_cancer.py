## **task 1:** split the data into a train and a test dataset

X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(X_cancer,
                                                                                y_cancer,
                                                                                random_state=0,
                                                                                stratify=y_cancer)


print("fraction of class malignant in train",sum(y_train_cancer)/len(y_train_cancer))
print("fraction of class malignant in test ",sum(y_test_cancer)/len(y_test_cancer) )
print("fraction of class malignant in full ",sum(y_cancer)/len(y_cancer))


## **task 2:** design your pipeline and grid search

pipeline_lr_cancer=Pipeline([('scaler',StandardScaler()),
                             ('model',LogisticRegression(solver = 'liblinear'))])

#define the hyper-parameter space to explore
grid_values = { 'model__C': np.logspace(-5,2,100),
               'model__penalty':['l1','l2'] }

#define the GridSearchCV object
grid_cancer = GridSearchCV( pipeline_lr_cancer, 
                           param_grid = grid_values,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1 )

#train your pipeline
grid_cancer.fit( X_train_cancer , y_train_cancer )


#get the best cross-validated score 
print(f'Grid best score ({grid_cancer.scoring}): {grid_cancer.best_score_:.3f}')
# print the best parameters
print('Grid best parameter :')
for k,v in grid_cancer.best_params_.items():
    print(' {:>20} : {}'.format(k,v))
