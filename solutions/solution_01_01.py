df=pd.read_csv("data/Human_nuisance.csv", index_col=0)
df = df.rename( columns = {"Breeding density(individuals per ha)" : "Breeding" , 
            "Number of pedestrians per ha per min" : "Number" } )

## setup a variable to the second and third degrees
df["Number2"] = df.Number ** 2
df["Number3"] = df.Number ** 3

## setup of the model - formula style
model = smf.ols( 'Breeding ~ Number  + Number2 + Number3' , data=df)# we create the least square fit object
results = model.fit()#we do the actual fit
print(results.summary())

LMstat , LMpval , Fstat , Fpval = het_white( results.resid , model.exog )
print("\n\tWhite test for heteroscedasticity p-value:" , LMpval)

## plotting results.
## the predict function will let us compute the prediction of the model from a single value of number of pedestrian
def predict( x ):
    return (results.params['Intercept'] + 
            x    *  results.params['Number'] +
            x**2 *  results.params['Number2'] +
            x**3 *  results.params['Number3'] )

xx=np.linspace(min( df.Number ),max( df.Number),200)

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(df.Number, df.Breeding, 'o', label="observed points")
ax.plot(df.Number, results.fittedvalues , 'or', label="fitted points")
ax.plot(xx, predict(xx), 'r')
ax.legend()
