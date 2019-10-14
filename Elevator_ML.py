#!/usr/bin/python
import time;
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt




localtime = time.localtime(time.time())
print ("Local current time :", localtime)

dataset = pd.read_csv('big_data5.csv')
size = dataset.shape
days=30; #days for random data
floors=int(size[1]/days); #apartment floor number
big_data=np.array([[[0]*days]*floors]*size[0])
time_var=1440/size[0]
flr=np.array([[0]*floors])
dates = list(range(1,days+1))
demandmatrix=np.array([[0]*floors]*1)
nextday=days+1
for i in range(1,floors+1):
    flr[:,i-1]=np.array([eval("button"+str(i))])

for i in range(days):
   big_data[:,range(floors-1),i]=(dataset.iloc[:,range((i+1)*floors-floors,(i+1)*floors-1)].values)

def predict_demand(dates, demands, x):
	dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1

	svr_rbf = SVR(kernel= 'rbf', C= 1e6, gamma= 100) 
	svr_lin = SVR(kernel= 'sigmoid', C= 1e3, coef0=1)
	svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
	svr_rbf.fit(dates, demands) 
	svr_lin.fit(dates, demands)
	svr_poly.fit(dates, demands)

#	plt.scatter(dates, demands, color= 'black', label= 'Data') # plotting the initial datapoints 
#	plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
#	plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
#	plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel'''
#	plt.xlabel('Day')
#	plt.ylabel('Demand Value')
#	plt.title('Deman Value Support Vector Regression Prediction')
#	#plt.legend()
#	plt.show()

	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]


d={}
dp={}
for i in range(0,floors):
    d["demand{0}".format(i)]=big_data[:,i,:]
    for k in range(0,size[0]):
        dp["demandparticular{0}".format(i)+str(k)]=list(d["demand{0}".format(i)][k,:])

'''for i in range(0,floors):
    for k in range(0,size[0]):
        predicted_demand=predict_demand(dates,dp["demandparticular{0}".format(i)+str(k)],nextday)
        demandmatrix[k,i,:]=predicted_demand[:]
        #for z in range(0,2):'''
def predict_dest(hour,minute):
        time_val = int(hour*60/time_var+int(np.rint(minute/time_var)));
        for i in range(0,floors):
            print(i)
            print("demandparticular{0}".format(i)+str(time_val))
            predicted_demand=predict_demand(dates,dp["demandparticular{0}".format(i)+str(time_val)],nextday)
            demandmatrix[:,i]=predicted_demand[0]
            print(predicted_demand[0])
        return demandmatrix.index(max(demandmatrix))
#predicted_demand=predict_demand(dates,dp["demandparticular689"],nextday)
#print ("\nThe predicted demand value for 7th floor between 7:30 and 7:35:")
#print ("RBF kernel: ", str(predicted_demand[0]))
#print ("Linear kernel: ", str(predicted_demand[1]))
#print ("Polynomial kernel: ", str(predicted_demand[2]))
