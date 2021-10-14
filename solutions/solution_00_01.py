import matplotlib.pyplot as plt 
import numpy as np

#1.
x = np.arange(-5,5,0.1)
y = 1/(1+np.exp(-x))
plt.plot(x,y)

#2.
for b in [0.5,1,2,4]:
    y2 = 1/(1+np.exp(-x*b))
    #plt.plot(x,y)
    plt.plot(x,y2 , label = 'b='+str(b) )
plt.legend()
plt.show()
