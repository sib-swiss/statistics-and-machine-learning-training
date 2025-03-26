# Remember that each tree in the forest only sees a random fraction of the features.
#
# As the number of "noise" features increases, the probability that any tree will get the combination of informative features  diminishes.
#
# Furthermore, the trees which see only noise also contribute some (uninformative) vote to the overall total.
#
# Thus it becomes harder to extract the signal from the noise in the data. 
#
#
# While this could be solved by increasing the number of trees. 
# It is often also advisable to perform some sort of **feature selection** to make sure you only present features of interest to the model.
#
# There are many procedures to do this, and none of these techniques are perfect however but, just to cite a few:
#
#  * select the X features which show the most differences between categories
#  * use PCA and limit yourself to the first few principal components
#  * use a reduced set of features externally defined with experts
#  * test random sets of features (but this is also very computationaly demanding)
#  * see the [feature selection page of sklearn](https://scikit-learn.org/stable/modules/feature_selection.html) 
#

## simple example with a selectKBest 
##  which will select the features with the highest ANOVA F-value between feature and target.
from sklearn.feature_selection import SelectKBest


ppl = Pipeline([('select' , SelectKBest( k = 100 ) ) , ## we will select 100 features, which is way to much here
                ('tree' , RandomForestClassifier(n_estimators=100))  ])

print("select 100 best > RF cross-validated accuracy:" , cross_val_score( ppl , 
                                                                         X_2_noise , 
                                                                         y_2 , 
                                                                         scoring='accuracy') )
