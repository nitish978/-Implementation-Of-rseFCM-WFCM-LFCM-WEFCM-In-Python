import pandas as pd
import numpy as np
import random
import operator
import math
df_full =pd.read_csv("Iris.csv")
columns = list(df_full.columns)
features = columns[1:len(columns)-1]
class_labels=list(df_full[columns[-1]])
df = df_full[features]
num_attr=len(features)
#cluster number
k=3
#max number of iterations
max_iter=1000
# Number of data points
n = len(df)
m = 1.80
def initializeMembershipMatrix(x):
  membership_mat = list()
  for i in range(len(x)):
   random_num_list = [random.random() for i in range(k)] 
   summation = sum(random_num_list)
   temp_list = [x1/summation for x1 in random_num_list]
   membership_mat.append(temp_list)
  return membership_mat
def initializeWeight(x):
    Weight= [1 for i in range(len(x))] 
    return Weight 

def calculateClusterCenter(membership_mat,W,sx):
     cluster_mem_val = zip(*membership_mat)   
     cluster_centers = list()
     for j in range(k):
     	x = list(cluster_mem_val[j])
     	xraised=[e ** m for e in x]
     	xraised_mul_W=[a*b for a,b in zip(xraised,W)]
     	denominator=sum(xraised_mul_W)
     	temp_num = list()
     	for i in range(len(sx)):
     	 data_point = sx[i]
         prod = [xraised_mul_W[i] * val for val in data_point]
         temp_num.append(prod)
        numerator = map(sum, zip(*temp_num)) 
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
     return cluster_centers
def updatemembershipvalue(U,C,sx):
 alpha=float(1/(m-1))
 for i in range(k):
  for j in range(len(sx)):
   x=sx[j]	
   numerator=[(a-b)**2 for a,b in zip(x,C[i])]
   num=sum(numerator)
   dis=[map(operator.sub,x,C[k1]) for k1 in range(k)]
   denominator=[map(lambda x: x**2, dis[j1]) for j1 in range(k)]
   den=[sum(denominator[k1]) for k1 in range(k)]
   res=sum([math.pow(float(num/den[k1]),alpha) for k1 in range(k)])
   U[j][i]=float(1/res)
 return U  
def updateweight(U,sx,w):
  W=list()
  u=zip(*U)
  for i in range(k):
   u1=sum([a*b for a,b in zip(u[i],w)])
   W.append(u1)	
  return W	
def WFCM(sx,W,U,C):
 i=0
 while(i<=max_iter):     
  U=updatemembershipvalue(U,C,sx)
  C=calculateClusterCenter(U,W,sx)
  i+=1
 return C,U      

def spFCM():
 X_sampled=list()	
 for j in range(3):
  l=len(df)/3;
  x=list() 	
  for i in range(j*l,min(j*l+l,len(df))):
   data=list(df.iloc[i])
   x.append(data)
  X_sampled.append(x)    
 W=initializeWeight(X_sampled[0])
 U=initializeMembershipMatrix(X_sampled[0])
 center=calculateClusterCenter(U,W,X_sampled[0])
 C,U=WFCM(X_sampled[0],W,U,center)
 X=X_sampled[0]
 for j in range(1,3):
  W1=updateweight(U,X,W)
  X=center+ X_sampled[j] 
  W2=initializeWeight(X)
  W=W1+W2
  U=initializeMembershipMatrix(X)
  center=calculateClusterCenter(U,W,X)
  C,U=WFCM(X,W,U,center)
 return C 
C=spFCM()
print C
