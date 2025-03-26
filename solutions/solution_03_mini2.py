reg = SGDRegressor( penalty='l1' , alpha = 10**bestLogAlpha  )
reg.fit( X , y )

y_pred = reg.predict( X )
print(f"train data R-squared score: { r2_score( y , y_pred ) :.2f}")
print(f"train data mean squared error: { mean_squared_error(  y , y_pred ) :.2f}")


y_test_pred = reg.predict( X_test )

print(f" test data R-squared score: { r2_score( y_test , y_test_pred ) :.2f}")
print(f" test data mean squared error: { mean_squared_error(  y_test , y_test_pred ) :.2f}")

plt.scatter( y , y_pred , label = 'training data' )
plt.scatter( y_test , y_test_pred , label = 'new data' )
plt.xlabel('observed values')
plt.ylabel('predicted values')
plt.legend()
