import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoLarsCV
import statistics
import time
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#parameters
Deg1=1#degree of polynomial we fit a(z)
Deg2=1#degree of polynomial we fit b(z)
supressPrint=1#if=0,will print nuisance estimators
policy=1#2 is 10% increase; 1 is original price

#load dataset
df = pd.read_csv('NomisB_e-Car_Data_3.csv')
sample_size = df.shape[0]
Z_train= np.array(df[['FICO','Amount','Cost of Funds',
                      'term72','term66','term60','term48',
                      'tier1','tier2','tier3',
                      'quarter1','quarter2','quarter3',
                      'year1','year2',
                      'partnerbin1','partnerbin2',]])
D_train = np.array(df[['Outcome']]).flatten()
P_train = np.array(df[['Rate']]).flatten()

#split into train and test
rndseed=252
P_train, P_test, Z_train,Z_test, D_train,D_test =train_test_split(
    P_train,Z_train,D_train, test_size=0.95,random_state=rndseed)
train_size=Z_train.shape[0]
test_size=Z_test.shape[0]

#the demand function
def demand_fcn(z,p):
    """This function takes in price p and feature z, and it returns demand"""
    return a_fcn(z)-b_fcn(z)*p

#the pricing function
def pricingFunction(p):
    if policy==1:
        return p#historical price
    if policy==2:
        return p*1.1#10% increase in price

#policy evaluation using doubly robust method
def main_evaluate_policy():
    #on training dataset
    print('on training dataset: ')
    TrainData=prepare_training_data()
    if policy==1:
        print('real value of original policy: ',TrainData.R_real,'\n')
    reg_result=direct_method_estimate(TrainData)
    a_b_est=doubly_robust_estimate(TrainData,reg_result)
    est_val=evaluate_policy(TrainData,a_b_est)
    real_rev=np.multiply(D_train,P_train)
    print('direct method: ')
    print('in-sample mean squared error: ', mean_squared_error(real_rev, est_val.R_DM))
    print('in-sample estimated value of given policy: ',est_val.V_DM)

    print('Victor\'s method:')
    print('in-sample mean squared error: ', mean_squared_error(real_rev, est_val.R_DR))
    print('in-sample estimated value of given policy: ',est_val.V_DR)
    
    print('logistic regression:')
    logReg_est=log_reg_estimate(TrainData)
    print('in-sample mean squared error: ', mean_squared_error(real_rev, logReg_est.revenue))
    print('in-sample estimated value of given policy: ',logReg_est.rev,'\n')
    #on testing dataset
    print('on testing dataset: ')
    TestData=prepare_testing_data()
    if policy==1:
        print('real value of original policy: ',TestData.R_real,'\n')
    a_b_est=doubly_robust_estimate(TestData,reg_result)
    est_val=evaluate_policy(TestData,a_b_est)
    real_rev=np.multiply(TestData.D_train,TestData.P_train)
    print('direct method: ')
    print('out-of-sample mean squared error: ', mean_squared_error(real_rev, est_val.R_DM))
    print('out-of-sample estimated value of given policy: ',est_val.V_DM)

    print('Victor\'s method:')
    print('out-of-sample mean squared error: ', mean_squared_error(real_rev, est_val.R_DR))
    print('out-of-sample estimated value of given policy: ',est_val.V_DR)
    
    print('logistic regression:')
    logReg_est=log_reg_estimate(TestData)
    print('out-of-sample mean squared error: ', mean_squared_error(real_rev, logReg_est.revenue))
    print('out-of-sample estimated value of given policy: ',logReg_est.rev)
    
#use logistic regression to evaluate policy
class log_reg_estimate: 
    def __init__(self,TrainData):
        X1=PolynomialFeatures(degree=Deg1, include_bias=True).fit_transform(TrainData.Z_train)
        X2=TrainData.P_train.reshape(-1,1)
        numCol1=np.shape(X1)[1]
        numCol2=np.shape(X2)[1]
        X=np.array([[0.0 for i in range(numCol1+numCol2)] for j in range (TrainData.train_size)]) 
        X[:,0:numCol1]=X1
        X[:,numCol1:]=-X2
        model = LogisticRegression(penalty='none',fit_intercept=False).fit(X, TrainData.D_train)
        HT=model.coef_
        #now predict value of new policy
        if policy==2:
            X2=1.1*X2
            X[:,numCol1:]=-X2
        pred_demand=model.predict_proba(X)[:,1]
        pred_rev=pred_demand@TrainData.P_train/TrainData.train_size
        if policy==2:
            pred_rev=1.1*pred_demand@TrainData.P_train/TrainData.train_size       
        if supressPrint==0:
            print('hat a: ',HT[0:numCol1])
            print('hat b: ',HT[numCol1:])
            
        self.HT=HT
        self.numCol1=numCol1
        self.numCol2=numCol2
        self.rev=pred_rev
        self.demand=pred_demand
        self.revenue=np.multiply(pred_demand,TrainData.P_train)
        
class prepare_training_data: 
    def __init__(self):
        self.P_train=P_train
        self.D_train=D_train
        self.Z_train=Z_train            
        self.R_real=P_train.dot(D_train)/train_size#real value of implemented pricing policy
        self.train_size=train_size
        
class prepare_testing_data: 
    def __init__(self):
        self.P_train=P_test
        self.D_train=D_test
        self.Z_train=Z_test            
        self.R_real=P_test.dot(D_test)/test_size#real value of implemented pricing policy
        self.train_size=test_size
        
class direct_method_estimate: 
    def __init__(self,TrainData):
        X1=PolynomialFeatures(degree=Deg1, include_bias=True).fit_transform(TrainData.Z_train)
        temp=PolynomialFeatures(degree=Deg2, include_bias=True).fit_transform(TrainData.Z_train)
        X2=np.multiply(TrainData.P_train.reshape(-1,1),temp)
        numCol1=np.shape(X1)[1]
        numCol2=np.shape(X2)[1]
        X=np.array([[0.0 for i in range(numCol1+numCol2)] for j in range (TrainData.train_size)]) 
        X[:,0:numCol1]=X1
        X[:,numCol1:]=-X2
        model = LassoLarsCV(fit_intercept=False).fit(X, TrainData.D_train)
        HT=model.coef_
        alpha=model.alpha_
        #print('alpha chosen by Lasso Cross Validation: ',alpha)
        if supressPrint==0:
            print('hat a: ',HT[0:numCol1])
            print('hat b: ',HT[numCol1:])
        model = LassoLarsCV(fit_intercept=False).fit(X1, TrainData.P_train)
        g=model.coef_
        model2=LassoLarsCV(fit_intercept=False).fit(X1, np.multiply(
            TrainData.P_train,TrainData.P_train))
        g2=model2.coef_
    
        self.HT=HT
        self.numCol1=numCol1
        self.numCol2=numCol2
        self.g=g
        self.g2=g2

class doubly_robust_estimate:
    def __init__(self,TrainData,reg_result):
        train_size=TrainData.train_size
        Xv=PolynomialFeatures(degree=1, include_bias=True).fit_transform(TrainData.Z_train)
        Xv2=PolynomialFeatures(degree=1, include_bias=True).fit_transform(TrainData.Z_train)
        a_DM=np.array([0.0]*train_size)
        b_DM=np.array([0.0]*train_size)
        a_DR=np.array([0.0]*train_size)
        b_DR=np.array([0.0]*train_size)
        res=np.array([0.0]*train_size)
        #estimate g and sigma
        g=reg_result.g
        g2=reg_result.g2
        g_z=Xv.dot(g)
        g2_z=Xv.dot(g2)
        varP=g2_z-np.multiply(g_z,g_z)
        
        for i in range(0,train_size):
            p=TrainData.P_train[i]
            A=[[1,-g_z[i]],[-g_z[i],g2_z[i]]]
            Ainv=np.linalg.pinv(A, rcond=1e-4)
            f=Ainv@[1,-p]
            
            a_DM[i]=reg_result.HT[0:reg_result.numCol1]@Xv[i]
            b_DM[i]=reg_result.HT[reg_result.numCol1:]@Xv2[i]
            res[i]=TrainData.D_train[i]-a_DM[i]+b_DM[i]*TrainData.P_train[i]
            
            #a_DR[i]=a_DM[i]+((g2_z[i]-g_z[i]*p)/varP[i])*res[i]                               
            #b_DR[i]=b_DM[i]+((g_z[i]-p)/varP[i])*res[i]
            a_DR[i]=a_DM[i]+f[0]*res[i]
            b_DR[i]=b_DM[i]+f[1]*res[i]
        #print('max g(z): ',max(g_z),' min g(z): ',min(g_z))
        print('min varP: ',min(abs(varP)),' avg varP: ',np.average(varP))
        counter=0
        for i in range(0,train_size):
            if b_DM[i]>0:
                counter=counter+1
        print('% of positive b(z): ',counter/TrainData.train_size)
              
        self.a_DM=a_DM
        self.b_DM=b_DM
        self.a_DR=a_DR
        self.b_DR=b_DR

class evaluate_policy:
    def __init__(self,TrainData,a_b_est):
        train_size=TrainData.train_size
        R_DM=np.array([0.0]*train_size)
        R_DR=np.array([0.0]*train_size)
        D_DM=np.array([0.0]*train_size)
        D_DR=np.array([0.0]*train_size)
        for i in range(0,train_size):
            p=pricingFunction(TrainData.P_train[i])
            D_DM[i]=a_b_est.a_DM[i]-a_b_est.b_DM[i]*p
            D_DR[i]=a_b_est.a_DR[i]-a_b_est.b_DR[i]*p
            R_DM[i]=p*D_DM[i]
            R_DR[i]=p*D_DR[i]

##        print('max D_DM: ',max(D_DM))
##        print('max D_DR: ',max(D_DR))
##        print('min D_DM: ',min(D_DM))
##        print('min D_DR: ',min(D_DR))
##        print('average D_DM: ',np.average(D_DM))
##        print('average D_DR: ',np.average(D_DR),'\n')
            
        if supressPrint==0:
            print('value of Pi(z) under DM estimators: ',np.average(R_DM))
            print('value of Pi(z) under DR estimators: ',np.average(R_DR))

        self.V_DM=np.average(R_DM)
        self.V_DR=np.average(R_DR)
        self.D_DR=D_DR
        self.D_DM=D_DM
        self.R_DM=R_DM
        self.R_DR=R_DR
    
main_evaluate_policy()
