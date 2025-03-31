X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(X_heart,y_heart,
                                                   random_state=123456,stratify=y_heart)
#stratify is here to make sure that you split keeping the repartition of labels unaffected

print("fraction of class benign in train",sum(y_train_heart)/len(y_train_heart))
print("fraction of class benign in test ",sum(y_test_heart)/len(y_test_heart))
print("fraction of class benign in full ",sum(y_heart)/len(y_heart))

### takes ~1minute to run

from sklearn.preprocessing import StandardScaler,PolynomialFeatures
# don't forget the scaler , 
# I also put a polynomial there, but with a twist : I will not really go for power 2,3,4..., 
#  but rather use it to create the interaction terms between the different features.,
pipeline_lr_heart=Pipeline([('scalar',StandardScaler()),
                            ('poly',PolynomialFeatures(include_bias=False , interaction_only=True)),
                            ('model',LogisticRegression(class_weight='balanced', solver = "liblinear"))])

grid_values = {'poly__degree':[1,2],
               'model__C': np.logspace(-5,2,100),
               'model__penalty':['l1','l2']}

grid_lr_heart = GridSearchCV(pipeline_lr_heart, 
                                     param_grid = grid_values, 
                                     scoring="balanced_accuracy",
                                     n_jobs=-1)

grid_lr_heart.fit(X_train_heart, y_train_heart)#train your pipeline

print('Grid best score ('+grid_lr_heart.scoring+'): ', 
      grid_lr_heart.best_score_)
print('Grid best parameter (max.'+grid_lr_heart.scoring+'): ', 
      grid_lr_heart.best_params_)

### takes ~1minute to run

# rbf kernel
pipeline_SVM_heart = Pipeline([('scalar',StandardScaler()),
                               ("classifier", SVC(class_weight='balanced', probability=True, kernel='rbf'))])
grid_values2 = {"classifier__gamma": np.logspace(-2,1,30)}
grid_svm_rbf_heart = GridSearchCV(pipeline_SVM_heart, grid_values2, 
                        n_jobs=-1, scoring="balanced_accuracy") 
grid_svm_rbf_heart.fit(X_train_heart, y_train_heart)
print('Grid best score ('+grid_svm_rbf_heart.scoring+'): ', 
      grid_svm_rbf_heart.best_score_)
print('Grid best parameter (max.'+grid_svm_rbf_heart.scoring+'): ', 
      grid_svm_rbf_heart.best_params_)

### takes ~30s to run

grid_values3 = {'criterion': ['entropy','gini'],
               'n_estimators':[100,250,500], 
               'max_depth':[10,15],
               'min_samples_split':[25,50],
              'min_samples_leaf':[10,25]}

grid_RF_heart = GridSearchCV(RandomForestClassifier(class_weight='balanced'), 
                              param_grid = grid_values3, 
                              scoring='balanced_accuracy',
                              n_jobs=-1)

grid_RF_heart.fit(X_train_heart, y_train_heart)

print('Grid best score ('+grid_RF_heart.scoring+'): ', grid_RF_heart.best_score_)
print('Grid best parameter (max. '+grid_RF_heart.scoring+'): ', grid_RF_heart.best_params_)


## the best model I have found is the logistic regression with polynomial of degree 2

## assess the performance of our fitted estimator on the test set
# calculate the score of your trained pipeline on the test
y_test_heart_scores = grid_lr_heart.score(X_test_heart,y_test_heart)

print('Grid best parameter (max.'+grid_lr_heart.scoring+') model on test: ', 
      y_test_heart_scores)

#predict y_test from X_test thanks to your trained model
y_test_heart_pred = grid_lr_heart.predict(X_test_heart)

# check the number of mistake made with the default threshold for your decision function
confusion_m_heart = confusion_matrix(y_test_heart, y_test_heart_pred)

plt.figure(figsize=(5,4))
sns.heatmap(confusion_m_heart, annot=True, fmt='d')
plt.title('LogReg degree : {}, C: {:.3f} , norm: {}\nAccuracy:{:.3f}'.format(
                    grid_lr_heart.best_params_['poly__degree'],
                    grid_lr_heart.best_params_['model__C'], 
                    grid_lr_heart.best_params_['model__penalty'] , 
                    accuracy_score(y_test_heart, y_test_heart_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')


## plotting the ROC curve of this model

# predict_proba gives you the proba for a point to be in both class
y_heart_score_lr = grid_lr_heart.predict_proba(X_test_heart)[:,1]
# compute the ROC curve:
fpr_heart, tpr_heart, threshold_heart = roc_curve(y_test_heart, 
                                                  y_heart_score_lr)
#finally this calculates the area under the curve
roc_auc_heart = auc(fpr_heart , tpr_heart )

#proba=sc.special.expit(thre_heart_roc_auc)
keep = np.argmin( abs(threshold_heart-0.5) ) # getting the theshold which is the closest to 0.5        

fig,ax = plt.subplots()
ax.set_xlim([-0.01, 1.00])
ax.set_ylim([-0.01, 1.01])
ax.plot(fpr_heart, tpr_heart, lw=3, label='LogRegr ROC curve\n (area = {:0.2f})'.format(roc_auc_heart))
ax.plot(fpr_heart[keep], tpr_heart[keep],'ro',label='threshold=0.5')
ax.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
ax.set_xlabel('False Positive Rate', fontsize=16)
ax.set_ylabel('True Positive Rate', fontsize=16)
ax.set_title('ROC curve (logistic classifier)', fontsize=16)
ax.legend(loc='lower right', fontsize=13)
ax.set_aspect('equal')


best_reg = grid_lr_heart.best_estimator_['model']
poly     = grid_lr_heart.best_estimator_['poly']

# each coefficient is a composite of the different columns, at different degree
# represented by a vector of powers
# exemple, with 4 features : X[:,0] ** 1 * X[:,3] ** 2
#                     --> [1,0,0,2]
coef_names = []
for i, row in enumerate( poly.powers_ ):
    n = []
    for j,p in enumerate(row):
        if p > 0:
            n.append(X_heart.columns[j])
            if p>1:
                n[-1] += "^"+str(p)
    coef_names.append("_x_".join(n) )

sorted_features=sorted( [(coef_names[i],abs(best_reg.coef_[0,i])) for i in range(len(poly.powers_))] ,
                       key=itemgetter(1),reverse=True)

print('Important features')

for feature, weight in sorted_features:
    if weight == 0: # ignore weight which are at 0
        continue
    print('\t{:>30}\t{:.3f}'.format(feature,weight) )
    

## finally, one little diagnostic plot which can sometimes be useful : 
## plot prediction probabilities of the correctly classified versus wrongly classified cases

## so we can see how much some positive case still get a very low probability, and vice version 

df = pd.DataFrame( {'y_true' : y_test_heart,
               'y_predicted' : y_test_heart_pred,
               'proba_class1' : y_heart_score_lr })


fig,ax = plt.subplots(figsize=(10,5))
sns.violinplot( x='y_true',
               y='proba_class1', 
               density_norm='count',
               data=df, ax=ax , cut=0)
ax.axhline(0.5 , color = 'black')
