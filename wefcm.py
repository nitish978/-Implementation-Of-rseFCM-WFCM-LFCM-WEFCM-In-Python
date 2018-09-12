import pandas as pd
import numpy as np
import random
import operator
import math

df_full =pd.read_csv("Iris.csv")
columns = list(df_full.columns)  #contain all columns name of data point
features = columns[1:len(columns)-1] # conatain all feature column name
class_labels=list(df_full[columns[-1]])
df = df_full[features]
#print features
num_attr=len(features)
#print num_attr
#number of clusters
k=3
#max number of iterations
max_iter=100

# Number of data points
n = len(df)

# Fuzzy parameter
m = 1.80
gamma=30

def initializeMembershipMatrix(n,k):
    membership_mat = list()
    for i in range(n):
     random_num_list = [random.random() for i in range(k)] 
     summation = sum(random_num_list)
     temp_list = [x/summation for x in random_num_list]
     membership_mat.append(temp_list)
    return membership_mat
def initializeWeight(num_attr,k):
  Weight=list()
  for i in range(k):
   random_num_list = [random.random() for i in range(num_attr)] 
   summation = sum(random_num_list)
   temp_list = [x/summation for x in random_num_list]
   Weight.append(temp_list)
  return Weight  
  

def calculateClusterCenter(membership_mat,k,X1,n):
    cluster_mem_val = zip(*membership_mat)
    cluster_centers = list()
    for j in range(k):
      x = list(cluster_mem_val[j])
      xraised = [e ** m for e in x]
      denominator = sum(xraised)
      temp_num = list()
      x=list()
      for i in range(len(X1)):
      	 x=X1[i]
         prod = [xraised[i] * val for val in x]
         temp_num.append(prod)
      numerator = map(sum,zip(*temp_num))
      center = [z/denominator for z in numerator]
      cluster_centers.append(center)
    return cluster_centers

def calculate_D1(Weight,cluster_center,n,k,z):
    D=list()
    for i in range(n):
     x=z[i]
     distance=[map(operator.sub,x,cluster_center[j]) for j in range(k)]
     dis=[map(lambda x: x**2, distance[j]) for j in range(k)]
     dis1=[map(operator.mul,Weight[j],dis[j]) for j in range(k)]
     dis2=[sum(dis1[j]) for j in range(k)]
     D.append(dis2)  
    return D

def updateMembershipValue(D,membership_mat,n,k):
   p = float(2/(m-1))
   for i in range(n):
    d=D[i]
    for j in range(k):
     #print d[j]	
     den=sum([math.pow(float(d[j]/d[k1]),p) for k1 in range(k)])  
     membership_mat[i][j]=float(1/den)
   return membership_mat
def calculate_d2(Weight,center,U,num_attr,n,k):
    D2=list() 
    u=zip(*U)
    #print len(u[0])
    for i in range(k):
     u_pow_alpha=[math.pow(x,m) for x in u[i]]
     dis=[map(operator.sub,df.iloc[j],center[i]) for j in range(n)]
     dis1=[map(lambda x: x**2, dis[j]) for j in range(n)]
     dis11=zip(*dis1)
     dis2=[map(lambda x,y: x*y,u_pow_alpha,dis11[j]) for j in range(num_attr)]
     dis3=[sum(dis2[x]) for x in range(num_attr)]
     D2.append(dis3)
    return D2 
def updateweight(D2,W,k,num_attr):
    for i in range(k):
     for j in range(num_attr):
      sum=0.0
      for k1 in range(num_attr):
       sum+=math.exp((-1*D2[i][k1])/gamma)
      W[i][j]=math.exp((-1*D2[i][j])/gamma)/sum
    return W
def getClusters(U,n):
  labels=list()
  for i in range(n):
   max_val, idx = max((val, idx) for (idx, val) in enumerate(U[i]))    
   labels.append(idx)
  return labels    
def WEFCM(z):
 num_attr=len(z[0])
 n=len(z) 
 W=initializeWeight(num_attr,k)
 U=initializeMembershipMatrix(n,k)
 center=calculateClusterCenter(U,k,z,n)
 i=0
 while(i<=max_iter):     
  D1=calculate_D1(W,center,n,k,z)
  U=updateMembershipValue(D1,U,n,k)
  D2=calculate_d2(W,center,U,num_attr,n,k)
  W=updateweight(D2,W,k,num_attr)
  center=calculateClusterCenter(U,k,z,n)
  i+=1
 return U,center
X=list()
for i in range(n):
 data=list(df.iloc[i])
 X.append(data)
U,c=WEFCM(X)
print "cluster c"
print c
#print U
#cluster_labels=getClusters(U)

