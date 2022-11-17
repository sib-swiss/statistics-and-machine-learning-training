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

##separation in X and y
X_heart = df_heart.drop( columns = "TenYearCHD" )
y_heart = df_heart[ "TenYearCHD" ]


X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(X_heart,y_heart,
                                                   random_state=0,stratify=y_heart)
#stratify is here to make sure that you split keeping the repartition of labels unaffected

print("fraction of class benign in train",sum(y_train_heart)/len(y_train_heart))
print("fraction of class benign in test ",sum(y_test_heart)/len(y_test_heart))
print("fraction of class benign in full ",sum(y_heart)/len(y_heart))


# don't forget the scaler 
pipeline_lr_heart=Pipeline([('scalar',StandardScaler()),
                            ('model',LogisticRegression(class_weight='balanced', solver = "liblinear"))])

grid_values = {'model__C': np.logspace(-5,2,100),
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
plt.title('LogReg C: {:.3f} , norm: {}\nAccuracy:{:.3f}'.format(grid_lr_heart.best_params_['model__C'], 
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

## getting the best features:
w_heart= np.abs( grid_lr_heart.best_estimator_[1].coef_[0] )

## get the weight order with argsort. We use flip to make the order descending
O = np.flip( np.argsort( w_heart  , ) )

## with zip we can combine 2 lists
sorted_features = list( zip( df_heart.columns[O] , w_heart[O] ) )

print('Features sorted per importance in discriminative process')
for f,w in sorted_features:
    print('{:>20}\t{:.3f}'.format(f,w))