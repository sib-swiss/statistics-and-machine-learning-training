from sklearn.linear_model import SGDRegressor

logalphas = []

coef_dict = {'name' : [],
             'coefficient' : [],
             'log-alpha' : []}
r2 = []

for alpha in np.logspace(-2,2,50):

    reg = SGDRegressor( penalty='l2' , alpha = alpha )
    reg.fit( X , y )
    
    logalphas.append(np.log10(alpha))
    r2.append( r2_score( y , reg.predict(X) ) )
    
    coef_dict['name'] += list( X.columns )
    coef_dict['coefficient'] += list( reg.coef_ )
    coef_dict['log-alpha'] += [np.log10(alpha)]* len(X.columns )

coef_df = pd.DataFrame(coef_dict)

fig,ax = plt.subplots(1,2,figsize = (14,7))

ax[0].plot(logalphas , r2)
ax[0].set_xlabel("log10( alpha )")
ax[0].set_ylabel("R2")

sns.lineplot( x = 'log-alpha' , y='coefficient' , hue = 'name' , data= coef_df , ax = ax[1] ,legend = False)

fig.suptitle("regression of potato data with an L2 regularization.")
