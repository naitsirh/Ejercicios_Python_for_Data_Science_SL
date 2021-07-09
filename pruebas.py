from math import sqrt
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'''
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
*  *  *  *  *  *  *  *   I N T R O D U C T I O N   *  *  *  *  *  *  *  
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
'''

# mean is the average value of the dataset
	# the mean value is usually not part of our dataset
prices = [18, 24, 67, 55, 42, 14, 19, 26, 33]

def mean(x):
	return sum(x) / len(x)

#print(mean(prices))



# median is the middle value of an ordered dataset
algo = [14, 18, 19, 24, 26, 33, 42, 55, 67, 68]
#        0   1   2   3   4   5   6   7   8   9
algo2 = [2,2,3,5,8,9]

def median(x):
	if len(x) % 2 == 1:
		return x[int(len(x)/2)]
	else:
		i = ( x[int(len(x)/2)] + x[int(len(x)/2) - 1] ) / 2
		if i % 2 != 0:
		 	return i
		else:
			return int(i)

#print(median(algo))



# variance is the average of the squared differences from the mean
algo3 = [14, 18, 19, 24, 26, 33, 42, 55, 67]

def variance(x):
	s = 0
	m = mean(x)
	for elem in x:
		s += (elem - m) ** 2
	return s / len(x)

#print(mean(algo3))
#print(variance(algo3))



# Standard Deviation is a measure of how spread out our data is
	# is the square root of the variance

def standard_deviation(x):
	return sqrt(variance(x))

#print(standard_deviation(algo3))



def std_devtn_exp(x):
	s = 0
	m = mean(x)
	for elem in x:
		s += abs(elem - m)
	return s / len(x)

#print(std_devtn_exp(algo3))



# elements within one standard deviation from the mean

def standards_deviations(x):
	s = 0
	for elem in x:
		if (elem > mean(x) - standard_deviation(x)) and (elem < mean(x) + 
			standard_deviation(x)):
			s += 1
	return s

#print(standards_deviations(algo3))


'''
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
*  *  *  *  *  *  *   MATH OPERATIONS WITH NUMPY   *  *  *  *  *  *  *  
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
'''


#x = np.array([1,2,3,4])
#print(x)
#print(x[0])



#x2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
#                |row|   |row|   |row|
#                | | |  <<--columns
#print(x2[1][2])

#print(x2.ndim)
# ndim returns the number of dimensions of the array

#print(x2.size)
# size returns the total number of elements of the array

#print(x2.shape)
# shape returns a tuple of integers that indicate the number of elements stored
	# along each dimension of the array
	# dimensions number (rows) >> elements stored (columns)



#x3 = np.array([2,1,3])

#x3 = np.append(x3, 4)
#np.append() add an element

#x3 = np.delete(x3, 2)
#np.delete() delete at index

#x3 = np.sort(x3)
#np.sort() sort the array

#print(x3)



#x4 = np.arange(2, 10, 3)
# np.arange() allows you to create an array that contains a range of evenly
	# spaced intervals (similar to a Python range)
#print(x4)



#x5 = np.arange(2, 8, 2)
#x5 = np.append(x5, x5.size)
#x5 = np.sort(x5)
#print(x5[1])



# shape refers to the number of rows and columns in the array

#x6 = np.arange(1, 7)
#print(x6)

#print(len(x6))

#print(list(range(7)))
#x7 = [1,2,3,4,5,6]
#print(x7)

#x8 = x6.reshape(3, 2)
# reshape() function allows us to change the shape of our arrays

#print(x8)



#x9 = np.arange(0, 20)
#print(x9)
#x10 = x9.reshape(5, 4)
#print(x10)



# reshape can also do the opposite: take a 2-dimensional array and make a
	# 1-dimensional array from it

#x11 = np.array([[1,2],[3,4],[5,6]])
#x12 = x11.reshape(6)
#print(x11)
#print(x12)



#x13 = np.arange(1, 8, 3)
#print(x13)
#x14 = x13.reshape(3, 1)
#print(x14)
#print(x14[1][0])



#x15 = np.arange(1, 10)
#print(x15)
#print(x15[0:2])
#print(x15[5:])
#print(x15[:2])
#print(x15[-1])
#print(x15[x15 < 4])
#print(x15[(x15 > 5) & (x15 % 2 == 0)])
#y = (x15 > 5) & (x15 % 2 == 0)
#print(x15[y])
#print(x15.sum())
#print(x15.min())
#print(x15.max())
x16 = [1,2,3,4,5,6,7,8,9]
#print(x16)
#print(x16 * 2)
#y = x16 * 2
#print(y)
#y2 = x15 * 2
#print(y2)



#x17 = np.array([14, 18, 19, 24, 26, 33, 42, 55, 67])
#x18 = np.array(algo3)

#print(x17)
#print(x18)

#print(np.mean(x17).round(2))
#print(np.median(x17))
#print(np.var(x17))
#print(np.std(x17).round(2))



#x19 = np.arange(3,9)
#y = x19.reshape(2,3)

#print(x19)
#print(y)


'''
data = np.array([150000, 125000, 320000, 540000, 200000, 120000, 160000, 
	230000, 280000, 290000, 300000, 500000, 420000, 100000, 150000, 280000])

#data_std = np.std(data)

#data_mean = np.mean(data)

def qtt_std(x):
	s = 0
	for elem in x:
		if (elem > np.mean(x) - np.std(x)) and (elem < np.mean(x) + np.std(x)):
			s += 1
	return s


def pcnt_std(x):
	return (qtt_std(x) / len(x)) * 100


#print(data_std)
#print(qtt_std(data))
#print(pcnt_std(data))
'''



'''
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
*  *  *  *  *  *  *   DATA MANIPULATION WITH PANDAS   *  *  *  *  *  *  
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
'''

'''
data = {
	'ages': [14, 18, 24, 42],
	'heights': [165, 180, 176, 184]
}

#df = pd.DataFrame(data)
#print(df)



x20 = {
	'a': [1,2],
	'b': [3,4],
	'c': [5,6]
}

df = pd.DataFrame(x20)
print(df)



df = pd.DataFrame(data, index=['James', 'Bob', 'Amy', 'Dave'])
print(df)
print()
#print(df.loc['Bob'])
#print(df['ages'])
#print(df[['ages', 'heights']])
#print(df.iloc[2])
#print(df.iloc[:2])
#print(df.iloc[1:3])
#print(df.iloc[-2:])
#print(df[(df['ages'] > 18) & (df['heights'] >= 180)])
'''

'''
df = pd.read_csv('ca-covid.csv')
#df = pd.read_csv("https://www.sololearn.com/uploads/ca-covid.csv")

#df.set_index('date', inplace=True)
df.drop('state', axis=1, inplace=True)

df['month'] = pd.to_datetime(df['date'], format='%d.%m.%y').dt.month_name()

df.set_index('date', inplace=True)

df['morb'] = df['deaths'] / df['cases'] * 100

#print(df.head())
#df.info()
#print(df.describe())
#print(df['cases'].describe())
#print(df['month'].value_counts())
#print(df.groupby('month')['cases'].sum())
#print(df.groupby('month')['morb'].mean())
#print((df.groupby('month')['deaths'].sum())/(df.groupby('month')['cases'].sum())*100)
print(df['cases'].sum())


data = {
	'height': [133,120,180,100],
	'age': [9,7,16,4]
}

df2 = pd.DataFrame(data)
print(df2['age'].mean())



data = {
	'a': [1,2,3],
	'b': [5,8,4]
}

df = pd.DataFrame(data)
df['c'] = df['a'] + df['b']

print(df.iloc[2]['c'])
'''

'''
df = pd.read_csv("ca-covid.csv")

df.drop('state', axis=1, inplace=True)
df.set_index('date', inplace=True)

df['ratio'] = df['deaths'] / df['cases']


#print(df)
#print(df['ratio'].max())
#print(df[df['ratio'] == 0.14285714285714285])
print(df[df['ratio'] == (df['ratio'].max())])
#print(df[(df['ages'] > 18) & (df['heights'] >= 180)])
'''



'''
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  
*  *  *  *  *  *  *   VISUALIZATION WITH MATPLOTLIB   *  *  *  *  *  *  
*  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  



s = pd.Series([18, 42, 9, 32, 81, 64, 3])

s.plot(kind='bar')
#			 barh
#			 hist
#			 box
#			 pie
#			 line
#plt.savefig('plot.png')
plt.show()


df = pd.read_csv('ca-covid.csv')
#df = pd.read_csv("https://www.sololearn.com/uploads/ca-covid.csv")

df.drop('state', axis=1, inplace=True)

df['date'] = pd.to_datetime(df['date'], format='%d.%m.%y')

df['month'] = df['date'].dt.month

#df['month'] = (df['date'].dt.month_name()).str[:3]

df['day'] = df['date'].dt.day

df.set_index('day', inplace=True)

#df.set_index('date', inplace=True)
'''



#df[df['month']==12]['cases'].plot()
#plt.savefig('plot2.png')
#plt.show()

#df[df['month']==8]['deaths'].plot()
#plt.show()

#df[df['month']==12][['cases', 'deaths']].plot()
#plt.savefig('plot3.png')
#plt.show()



#(df.groupby('month')['cases'].sum()).plot(kind='bar')
#plt.savefig('plot4.jpg')
#plt.show()



#df = df.groupby('month')[['cases', 'deaths']].sum()
#df.plot(kind='bar', stacked=True)
#plt.show()


'''
data = {
	'height': [133,120,180,100,170, 161],
	'age': [9,7,16,4,15,13]
}

df = pd.DataFrame(data)
df.set_index('age', inplace=True)
#print(df)

df = df.groupby('age')['height'].mean()
df.plot(kind='bar')
plt.show()
'''


#df[df['month']==6]['cases'].plot(kind='box')
#plt.savefig('plot6.png')
#plt.show()
#print(df[df["month"]==6].describe())



#df = pd.DataFrame({'values':[-12000000,10,15,20,98,45,100000,518388]})
#df.plot(kind="box")
#plt.show()



#df[df['month']==6]['cases'].plot(kind='hist', bins=20)
#plt.show()



#df[df['month']==6][['cases', 'deaths']].plot(kind='area', stacked=False)
#plt.savefig('plot7.png')
#plt.show()



#df[df['month']==6][['cases', 'deaths']].plot(kind='scatter', x='cases', 
#	y='deaths')
#plt.savefig('plot8.png')
#plt.show()



#df.groupby('month')['cases'].sum().plot(kind='pie')
#plt.savefig('plot9.png')
#plt.show()



#df = df[df['month']==6]

#df[['cases', 'deaths']].plot(kind='line', legend=True)
#plt.xlabel('Days in June')
#plt.ylabel('Number')
#plt.suptitle('COVID-19 in June')
#plt.savefig('plot11.png')
#plt.show()


'''
df[['cases', 'deaths']].plot(kind='area', 
	legend=True, 
	stacked=False, 
	color=['#1970E7', '#E73E19'])
plt.xlabel('Days in June')
plt.ylabel('Number')
plt.suptitle('COVID-19 in June')
#plt.savefig('plot12.png')
plt.show()
'''


data = {
	'sport':['Soccer', 'Tennis', 'Soccer', 'Hockey'],
	'players':[5,4,8,20]
}

df = pd.DataFrame(data)

df.groupby('sport')['players'].sum().plot(kind='pie')
plt.show()