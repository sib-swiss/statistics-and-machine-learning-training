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

# creating input for many different number of pedestrians in order to plot the curve
xx=pd.DataFrame( {"Number" : np.linspace(min( df.Number ),max( df.Number),200) })
xx["Number2"] = xx["Number"]**2
xx["Number3"] = xx["Number"]**3

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(df.Number, df.Breeding, 'o', label="observed points")
ax.plot(df.Number, results.fittedvalues , 'or', label="fitted points")
ax.plot(xx.Number, results.predict(xx), 'r')
ax.legend()
