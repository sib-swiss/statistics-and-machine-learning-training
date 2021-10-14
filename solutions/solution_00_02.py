import pandas as pd
import matplotlib.pyplot as plt

#1.
df = pd.read_table('data/kyphosis.csv',sep=',')
print( df.head() )

#2.
print( 'column number:', df.shape[1] )

#3.
print( 'maximum age:' , df['Age'].max() )

#4.
df['Stop'] = df['Start'] + df['Number']

#5.
# simple version
plt.plot( df['Age'] , df['Number'] , linestyle='',marker='v' )
plt.xlabel('age')
plt.ylabel('number')
plt.show()

# bonus version
maskKyphosis = df['Kyphosis'] == 'present' # the mask is a list of True/False values 

# by applying the mask to columns, we only keep the values we want
plt.plot( df['Age'][maskKyphosis] , 
          df['Number'][maskKyphosis] , 
          linestyle='',marker='v' , label='Kyphosis' ) 


plt.plot( df['Age'][maskKyphosis == False] , 
          df['Number'][maskKyphosis == False] , 
          linestyle='',marker='v' , label='absent' )# I invert the mask by asking for False values

plt.xlabel('age')
plt.ylabel('number')
plt.legend()
plt.show()
