import numpy as np
import math
import random
import pandas as pd
import operator 
from sklearn.metrics.pairwise import rbf_kernel
gamma1=1
m = 1.80
D=25
max_iter=100
df_full =pd.read_csv("SPECTF_New.csv")
columns = list(df_full.columns)
features = columns[1:len(columns)-1]
class_labels=list(df_full[columns[-1]])
df = df_full[features]
num_attr=len(features)
d11=num_attr
n=len(df)
D11=25
k=2
m = 1.80
gamma=30
num_attr=D11
def randomize_feature_map(W,b):
 X=list()
 for i in range(n):
  data=df.iloc[i]
  x=[np.dot(data,W[k1]) for k1 in range(D11)]
  #x3=[map(operator.sum,x,b)]
  x1=[np.cos(x[j]) for j in range(D11)]
  x2=[xx1/np.sqrt(D) for xx1 in x1]
  X.append(x2)
 return X

def initializeMembershipMatrix():
    membership_mat = list()
    for i in range(n):
     random_num_list = [random.random() for i in range(k)] 
     summation = sum(random_num_list)
     temp_list = [x/summation for x in random_num_list]
     membership_mat.append(temp_list)
    return membership_mat

def initializeWeight():
  Weight=list()
  for i in range(k):
   random_num_list = [random.random() for i in range(num_attr)] 
   summation = sum(random_num_list)
   temp_list = [x/summation for x in random_num_list]
   Weight.append(temp_list)
  return Weight
def calculateClusterCenter(membership_mat,X):
    cluster_mem_val = zip(*membership_mat)
    cluster_centers = list()
    for j in range(k):
      u= list(cluster_mem_val[j])
      xraised = [e ** m for e in u]
      denominator = sum(xraised)
      temp_num = list()
      x=list()
      for i in range(n):
      	 x=X[i]
      	# print x
         prod = [xraised[i] * val for val in x]
         temp_num.append(prod)
      numerator = map(sum,zip(*temp_num))

      center = [z/denominator for z in numerator]
      cluster_centers.append(center)
    return cluster_centers

def calculate_D1(Weight,cluster_center,X):
    D=list()
    for i in range(n):
     x=X[i]
     distance=[map(operator.sub,x,cluster_center[j]) for j in range(k)]
     dis=[map(lambda x: x**2, distance[j]) for j in range(k)]
     dis1=[map(operator.mul,Weight[j],dis[j]) for j in range(k)]
     dis2=[sum(dis1[j]) for j in range(k)]
     D.append(dis2)  
    return D
def updateMembershipValue(D,membership_mat):
   p = float(2/(m-1))
   for i in range(n):
    d=D[i]
    for j in range(k):	
     den=sum([math.pow(float(d[j]/d[k1]),p) for k1 in range(k)])  
     membership_mat[i][j]=float(1/den)
   return membership_mat

def calculate_d2(Weight,center,U,X):
    D2=list() 
    u=zip(*U)
    #print len(u[0])
    for i in range(k):
     u_pow_alpha=[math.pow(x,m) for x in u[i]]
     dis=[map(operator.sub,X[i],center[i]) for j in range(n)]
     dis1=[map(lambda x: x**2, dis[j]) for j in range(n)]
     dis11=zip(*dis1)
     dis2=[map(lambda x,y: x*y,u_pow_alpha,dis11[j]) for j in range(num_attr)]
     dis3=[sum(dis2[x]) for x in range(num_attr)]
     D2.append(dis3)
    return D2  

def updateweight(D2,W):
    for i in range(k):
     for j in range(num_attr):
      sum=0.0
      for k1 in range(num_attr):
       sum+=math.exp((-1*D2[i][k1])/gamma)
      W[i][j]=math.exp((-1*D2[i][j])/gamma)/sum
    return W
def getClusters(U):
  labels=list()
  for i in range(n):
   max_val, idx = max((val, idx) for (idx, val) in enumerate(U[i]))    
   labels.append(idx)
  return labels
def WEFCM():
 W1=np.sqrt(2*gamma1)*np.random.normal(size=(D11,d11))
 b=2*np.pi*np.random.rand(D11)
 X=randomize_feature_map(W1,b)
 W=initializeWeight()
 U=initializeMembershipMatrix()
 center=calculateClusterCenter(U,X)
 i=0
 while(i<=max_iter):     
  D1=calculate_D1(W,center,X)
  U=updateMembershipValue(D1,U)
  D2=calculate_d2(W,center,U,X)
  W=updateweight(D2,W)
  center=calculateClusterCenter(U,X)
  i+=1
 return U


U=WEFCM()
print "cluster c"
print U 
labels=getClusters(U)
print np.transpose(labels)
