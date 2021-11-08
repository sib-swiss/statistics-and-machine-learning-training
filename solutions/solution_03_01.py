import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve , auc , classification_report , accuracy_score , confusion_matrix

# reading the data
df_heart=pd.read_csv('data/framingham.csv')
#df_heart.replace(np.nan,"NaN")
df_heart.dropna(axis=0,inplace=True)

df_heart.head()

##separation in train/test sets
X_heart=np.array(df_heart[list(df_heart.columns)[:-1]])
y_heart=np.array(df_heart[list(df_heart.columns)[-1]])
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(X_heart,y_heart,
                                                   random_state=0,stratify=y_heart)
#stratify is here to make sure that you split keeping the repartition of labels unaffected

print("fraction of class benign in train",sum(y_train_heart)/len(y_train_heart),"fraction of class benign in test",sum(y_test_heart)/len(y_test_heart),"fraction of class benign in full",sum(y_heart)/len(y_heart))




## creating a pipeline to explore our hyper-parameters space and fit it
sco='roc_auc'
#create your logistic regression object, the class being slightly unbalanced add a class weight
logi_heart=LogisticRegression(class_weight='balanced')

# Put it in a pipeline : the pipeline allows you to put tasks to perfom in a sequential manner.
# Here particularly to scale subset of your data at a time when you will use the cross validation technique. 
# By doing the scaling on each subset that is going for validation instead of on the full training set
# you ensure that information about your test and validation are not leaking in your training.
# Scaling is important for some optimizers, generally speaking for technics other than logistic
# regression or decision tree, when you add a Lasso or Ridge regularization,
# when dealing with covariables that have a variety of scales, and finally I believe make model intepretation easier.

pipeline_lr_heart=Pipeline([('scalar',StandardScaler()),('model',logi_heart)])

# define the hyperparameters you want to test
#with the range over which you want it to be tested. Note the model double underscore name of the parameters.
grid_values = {'model__C': np.logspace(-5,2,100),'model__penalty':['l1','l2'],'model__solver':['liblinear']}

#Feed it to the GridSearchCV with the right
#score(here sc) over which the decision should be taken
grid_lr_heart_roc_auc = GridSearchCV(pipeline_lr_heart, 
                           param_grid = grid_values, 
                           scoring=sco,n_jobs=-1)

grid_lr_heart_roc_auc.fit(X_train_heart, y_train_heart)#train your pipeline

print('Grid best parameter (max.'+sco+'): ', grid_lr_heart_roc_auc.best_params_)#get the best parameters
print('Grid best score ('+sco+'): ', grid_lr_heart_roc_auc.best_score_)#get the best score calculated from the train/validation
#dataset



## assess the performance of our fitted estimator on the test set
y_heart_decision_fn_scores_roc_auc=grid_lr_heart_roc_auc.score(X_test_heart,y_test_heart)# calculate the score of your trained pipeline on the test

print('Grid best parameter (max.'+sco+') model on test: ', y_heart_decision_fn_scores_roc_auc)# get the equivalent score on the test
#dataset : again this is the important metric

y_heart_pred_test_roc_auc=grid_lr_heart_roc_auc.predict(X_test_heart)#predict y_test from X_test thanks to your trained model

confusion_m_heart = confusion_matrix(y_test_heart, y_heart_pred_test_roc_auc)# check the number of mistake made with the default 
#threshold for your decision function
print("confusion matrix")
print(confusion_m_heart)
df_c_heart = pd.DataFrame(confusion_m_heart, 
                     index = [i for i in range(2)], columns = [i for i in range(2)])

plt.figure(figsize=(5,4))
sns.heatmap(df_c_heart, annot=True, fmt='d')
plt.title('LogReg C: {:.3f} , norm: {}\nAccuracy:{:.3f}'.format(grid_lr_heart_roc_auc.best_params_['model__C'], 
                                                                 grid_lr_heart_roc_auc.best_params_['model__penalty'] , 
                                                                 accuracy_score(y_test_heart, y_heart_pred_test_roc_auc)))
plt.ylabel('True label')
plt.xlabel('Predicted label')

print(classification_report(y_test_heart, y_heart_pred_test_roc_auc))# check the overall capacity of your model on test set 
#according to a bunch of metric

## plotting the ROC curve of this model
# this three lines here are how you get the area under the ROC curve score which is very important for evaluating your model
y_heart_score_lr_roc_auc = grid_lr_heart_roc_auc.decision_function(X_test_heart)#decision_function gives you the proba for a point to be in
# a class
fpr_lr_heart_roc_auc, tpr_lr_heart_roc_auc, thre_heart_roc_auc = roc_curve(y_test_heart, y_heart_score_lr_roc_auc)# this calculates the ROC curve
roc_auc_lr_heart_roc_auc = auc(fpr_lr_heart_roc_auc, tpr_lr_heart_roc_auc)#finally this calculates the area under the curve

print(roc_auc_lr_heart_roc_auc)

proba=sc.special.expit(thre_heart_roc_auc)
keep = np.argmin( abs(proba-0.5) ) # getting the theshold which is the closest to 0.5        

fig,ax = plt.subplots()
ax.set_xlim([-0.01, 1.00])
ax.set_ylim([-0.01, 1.01])
ax.plot(fpr_lr_heart_roc_auc, tpr_lr_heart_roc_auc, lw=3, label='LogRegr ROC curve\n (area = {:0.2f})'.format(roc_auc_lr_heart_roc_auc))
ax.plot(fpr_lr_heart_roc_auc[keep], tpr_lr_heart_roc_auc[keep],'ro',label='threshold=0.5')
ax.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
ax.set_xlabel('False Positive Rate', fontsize=16)
ax.set_ylabel('True Positive Rate', fontsize=16)
ax.set_title('ROC curve (logistic classifier)', fontsize=16)
ax.legend(loc='lower right', fontsize=13)
ax.set_aspect('equal')

## getting the best features:
w_heart_roc_auc=grid_lr_heart_roc_auc.best_estimator_[1].coef_[0]

sorted_features=sorted([[df_heart.columns[i],abs(w_heart_roc_auc[i])] for i in range(len(w_heart_roc_auc))],key=itemgetter(1),reverse=True)

print('Features sorted per importance in discriminative process')
for f,w in sorted_features:
    print('{:>20}\t{:.3f}'.format(f,w))