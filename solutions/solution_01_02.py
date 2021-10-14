df=pd.read_csv("data/Human_nuisance.csv", index_col=0)
df = df.rename( columns = {"Breeding density(individuals per ha)" : "Breeding" , 
            "Number of pedestrians per ha per min" : "Number" } )


## let's build a reference model with only the number 

model = smf.ols( 'Breeding ~ Number' , data=df)
results = model.fit()

ref_model = 'Breeding ~ Number'
ref_results = results

# incrementally adding columns of increasing power
maxPow = 10
threshold = 0.05
coVariables = ['Number']
for current_power in range( 2,maxPow ) :

    # create the column we need
    df[ 'Number'+str(current_power) ] = df.Number ** current_power
    
    coVariables.append('Number'+str(current_power))
    
    formula = 'Breeding ~ ' + '+'.join(coVariables)
    
    model = smf.ols( formula , data=df)
    results = model.fit()
    
    # LRT test
    LRT = 2*(results.llf - ref_results.llf)
    pval = 1-stats.chi2.cdf( LRT , 1 )
    
    print( '{:>20} -> pval: {:.3f}'.format(formula,pval) )
    
    if pval < threshold :
        ## switch the reference 
        ref_model = formula
        ref_results = results
    
    else:
        break ## stop here

print("Best model:",ref_model)
print(ref_results.summary())
    