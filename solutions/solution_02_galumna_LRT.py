accepted_covars = [ 'WatrCont' ]
current_model = smf.glm( "Galumna ~ " + '+'.join(accepted_covars), data= mites,
               family=sm.families.Poisson( link = sm.families.links.Log() )).fit()
new_covars = [ 'SubsDens', 'Substrate', 'Topo']


pvals = []
for covar in new_covars:
    formula = "Galumna ~ " + '+'.join(accepted_covars) + '+' + covar
    modelPoisson2 = smf.glm( formula, data= mites,
               family=sm.families.Poisson( link = sm.families.links.Log() ))# family=Poisson link=log
    modelPoisson2 = modelPoisson2.fit()#we do the actual fit

    logLkhDiff =  2*(modelPoisson2.llf - current_model.llf)
    pval = 1-stats.chi2.cdf( logLkhDiff ,df=1)

    pvals.append(pval)

bestI = np.argmin(pvals)
best_pvalue = pvals[bestI]
best_covar = new_covars[bestI]
print( f"the best covariate to add is {best_covar} (LRT p-value:{best_pvalue:.1e})" )
modelPoisson2 = smf.glm( "Galumna ~ WatrCont + Substrate", data= mites,
               family=sm.families.Poisson( link = sm.families.links.Log() ))# family=Poisson link=log
modelPoisson2 = modelPoisson2.fit()#we do the actual fit
print(modelPoisson2.summary())

## looking at pearson residuals
fig,ax = plt.subplots( ncols=2, figsize = (14,5) )

ax[0].scatter( modelPoisson2.mu , modelPoisson2.resid_pearson )
ax[0].set_xlabel('predicted values')
ax[0].set_ylabel('pearson residuals')
ax[0].grid(axis='y')

ax[1].plot(modelPoisson2.mu , mites.Galumna , 'bo')
ax[1].plot( [0,max(mites.Galumna)], [0,max(mites.Galumna)] )
ax[1].set_xlabel("predicted values")
ax[1].set_ylabel("observed values")


LMstat , LMpval , Fstat , Fpval = het_white( modelPoisson2.resid_pearson , 
                                             pd.DataFrame({'const':1, 'x':modelPoisson2.mu})
                                           )
print("\n\tWhite test for heteroscedasticity p-value on the Pearson's residuals:" , LMpval)
