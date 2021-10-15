import numpy as np

# answer : the doubling time corresponds to the number of days d, 
# n0*exp(beta*d) = 2*n0*exp(beta*0)
# --> exp(beta*d) = 2
# --> beta*d == log(2)
doubling_time = np.log(2)/0.1128
doubling_time