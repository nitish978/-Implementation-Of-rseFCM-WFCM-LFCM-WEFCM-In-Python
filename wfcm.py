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
max_iter=100
# Number of data points
n = len(df)
m = 1.80
def initializeMembershipMatrix():
  membership_mat = list()
  for i in range(n):
   random_num_list = [random.random() for i in range(k)] 
   summation = sum(random_num_list)
   temp_list = [x/summation for x in random_num_list]
   membership_mat.append(temp_list)
  return membership_mat
def initializeWeight():
    Weight= [1 for i in range(n)] 
    return Weight 

def calculateClusterCenter(membership_mat,W):
     cluster_mem_val = zip(*membership_mat)   
     cluster_centers = list()
     for j in range(k):
     	x = list(cluster_mem_val[j])
     	xraised=[e ** m for e in x]
     	xraised_mul_W=[a*b for a,b in zip(xraised,W)]
     	denominator=sum(xraised_mul_W)
     	temp_num = list()
     	for i in range(n):
     	 data_point = list(df.iloc[i])
         prod = [xraised_mul_W[i] * val for val in data_point]
         temp_num.append(prod)
        numerator = map(sum, zip(*temp_num)) 
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
     return cluster_centers
def updatemembershipvalue(U,C):
 alpha=float(1/(m-1))
 for i in range(k):
  for j in range(n):
   x=list(df.iloc[j])	
   numerator=[(a-b)**2 for a,b in zip(x,C[i])]
   num=sum(numerator)
   dis=[map(operator.sub,x,C[k1]) for k1 in range(k)]
   denominator=[map(lambda x: x**2, dis[j1]) for j1 in range(k)]
   den=[sum(denominator[k1]) for k1 in range(k)]
   res=sum([math.pow(float(num/den[k1]),alpha) for k1 in range(k)])
   U[j][i]=float(1/res)
 return U  

def WFCM():
 W=initializeWeight()
 U=initializeMembershipMatrix()
 center=calculateClusterCenter(U,W)
 i=0
 while(i<=max_iter):     
  U=updatemembershipvalue(U,center)
  center=calculateClusterCenter(U,W)
  i+=1
 return center,U      

C,U=WFCM()
print C
