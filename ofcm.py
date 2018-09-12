import pandas as pd
import numpy as np
import random
import operator
import math
df_full =pd.read_csv("SUSY.csv")
columns = list(df_full.columns)
features = columns[1:len(columns)]
class_labels=list(df_full[columns[0]])
df = df_full[features]
num_attr=len(features)
#cluster number
k=2
#max number of iterations
max_iter=100
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
def updateweight(U1,c):
  W=list()
  for j in range(c):
   w1=list()
   u=zip(*U1[j])
   for i in range(k):
    u1=sum(u[i])
    w1.append(u1)
   W.append(w1) 	
  W1=list()
  W1=W[0]
  for j in range(1,len(W)):
   W1=W1+W[j]
  return W1  	
def WFCM(sx,W,U,C):
 i=0
 while(i<=max_iter):     
  U=updatemembershipvalue(U,C,sx)
  C=calculateClusterCenter(U,W,sx)
  i+=1
 return C,U      

def oFCM():
 X_sampled=list()	
 for j in range(5000):
  l=len(df)/5000;
  x=list() 	
  for i in range(j*l,min(j*l+l,len(df))):
   data=list(df.iloc[i])
   x.append(data)
  X_sampled.append(x)
 U1=list()
 C1=list()     
 W=initializeWeight(X_sampled[0])
 U=initializeMembershipMatrix(X_sampled[0])
 center=calculateClusterCenter(U,W,X_sampled[0])
 C,U=WFCM(X_sampled[0],W,U,center)
 U1.append(U)
 C1.append(C)
 for j in range(1,5000):
  X=X_sampled[j] 
  W=initializeWeight(X)
  U=initializeMembershipMatrix(X)
  U=updatemembershipvalue(U,C1[j-1],X)
  center=calculateClusterCenter(U,W,X)
  C,U=WFCM(X,W,U,center)
  U1.append(U)
  C1.append(C)
 W=updateweight(U1,3)
 C=C1[0]
 for j in range(1,len(C1)):
  C=C+C1[j]
 U=initializeMembershipMatrix(C)
 center=calculateClusterCenter(U,W,C)
 C,U=WFCM(C,W,U,center)
 return C 
C=oFCM()
print C
print len(df)
