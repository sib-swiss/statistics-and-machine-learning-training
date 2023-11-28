import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

df_beetles=pd.read_csv('data/beetle.csv' , index_col=0)
df_beetles

# setting up the model
#since we want our random variable to be represented by a binomial we need two parameters
#to define a binomial.
model = smf.glm("ndied+nalive ~ dose", data=df_beetles,
                family=sm.families.Binomial() )
results = model.fit()

print( results.summary() )

# adding a couple of diagnostic plots
fig, ax = plt.subplots(ncols=2,figsize=(14,6))
ax[0].plot(results.mu , results.resid_pearson , 'bo')
ax[0].set_xlabel("predicted values")
ax[0].set_ylabel("pearson residuals")


# results.mu corresponds to the fitted proportion of dead beetle
# equivalent to : 1./(1+np.exp(-(-14.8230+0.2494*df_beetles["dose"]))) 
ax[1].plot(results.mu , df_beetles["prop"] , 'bo')
ax[1].plot( [0,max(df_beetles["prop"])], [0,max(df_beetles["prop"])] )
ax[1].set_xlabel("predicted values")
ax[1].set_ylabel("observed values")

plt.tight_layout()

# plotting the prediction
predictedProportionOfDead =  results.mu 
# equivalent to : 1./(1+np.exp(-(-14.8230+0.2494*df_beetles["dose"]))) 

fig, ax = plt.subplots(ncols=1,figsize=(6,6))
ax.plot(df_beetles["dose"] , df_beetles["prop"] ,'bo',label='data')
ax.plot(df_beetles["dose"], predictedProportionOfDead,'r--',label='prediction GLM')

ax.set_xlabel('dose')
ax.set_ylabel('proportion of dead')
ax.legend(loc='best',fontsize=12)