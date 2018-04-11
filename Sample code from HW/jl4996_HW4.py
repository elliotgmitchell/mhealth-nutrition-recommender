#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 23:08:08 2018

@author: jeremylew
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Problem 2
nUsers=943
nMovies=1682

ratingsCSV = pd.read_csv("./hw4-data/ratings.csv",header=None)     
ratingsCSV.columns = ["user_id", "movie_id", "rating"]
ratingsMatrix = np.empty((nUsers,nMovies))*np.nan #943 users and 1682 movies
ratingsIndicator = np.zeros((nUsers,nMovies))  

ratingsTest = pd.read_csv("./hw4-data/ratings_test.csv",header=None)
ratingsTest.columns = ["user_id", "movie_id", "rating"]

# Populate the ratings matrix and ratings indicator matrix 
for row in range(0,ratingsCSV.shape[0]):
    ratingsIndicator[ratingsCSV.loc[row,"user_id"]-1,ratingsCSV.loc[row,"movie_id"]-1] = 1
    ratingsMatrix[ratingsCSV.loc[row,"user_id"]-1,ratingsCSV.loc[row,"movie_id"]-1] = ratingsCSV.loc[row,"rating"]

# Parameters
sigmasq = 0.25
d = 10
lambd = 1

def PredictOnTestSet(ratingsTest,U,V):
    predictedRatings = np.array([])
    for row in range(0,ratingsTest.shape[0]):
        predictedRatings = np.append(predictedRatings,
                                     np.dot(U[ratingsTest.loc[row,"user_id"]-1,:], 
                                            V[:,ratingsTest.loc[row,"movie_id"]-1]))
    return predictedRatings


def calculateRSME(predicted,test):
    if len(predicted) != len(test): raise Exception("Vector lengths not equal")
    else: return np.sqrt((1/len(test))*np.sum((test-predicted)**2))

    
def logJointLikelihood(U,V,ratingsMatrix,ratingsIndicator,sigmasq,lambd,nUsers,nMovies,d): 
    L = - np.sum([(1/(2*sigmasq))*
               (ratingsMatrix[row,column]-np.dot(U[row,:],V[:,column]))**2 
                for (row,column) in list(zip(*np.where(ratingsIndicator==1)))]) \
                - (lambd/2)*np.matrix.trace(np.dot(U,U.T)) \
                - (lambd/2)*np.matrix.trace(np.dot(V.T,V)) \
                - len(np.where(ratingsIndicator==1)[0]) * np.log(np.sqrt((2*np.pi*sigmasq))) \
                - nUsers * np.log((2*np.pi/lambd)**(d/2)) \
                - nMovies * np.log((2*np.pi/lambd)**(d/2))
    return(L)


def MAPMatrixCompletion(d,sigmasq,lambd,nMovies,nUsers,ratingsIndicator,ratingsMatrix,ratingsTest):
    #initialize all N1 Ui and N2 vj
    V=np.random.multivariate_normal(mean=np.zeros(d),cov=(1/lambd)*np.eye(d),size=nMovies).T
    U=np.random.multivariate_normal(mean=np.zeros(d),cov=(1/lambd)*np.eye(d),size=nUsers)
    logJointLikelihoodObj = np.array([])
    
    for t in range(0,100):
        
        #Update user location
        for i in range(0,nUsers):
            ratedObjIndexes = np.where(ratingsIndicator[i,:]==1)[0]
            Vrated = V[:,ratedObjIndexes]
            Mrated = ratingsMatrix[i,ratedObjIndexes]
            Vsum = np.dot(Vrated,Vrated.T)
            U[i,:]=np.dot(np.linalg.inv(lambd*sigmasq*np.eye(d) + Vsum), np.dot(Vrated,Mrated.T))
        
        #Update object location
        for j in range(0,nMovies):
            ratedUserIndexes = np.where(ratingsIndicator[:,j]==1)[0]
            Urated = U[ratedUserIndexes,:]
            Mrated = ratingsMatrix[ratedUserIndexes,j]           
            Usum = np.dot(Urated.T,Urated)            
            V[:,j] = np.dot(np.linalg.inv(lambd*sigmasq*np.eye(d) + Usum), np.dot(Urated.T,Mrated))
    
        logJL = logJointLikelihood(U,V,ratingsMatrix,ratingsIndicator,sigmasq,lambd,nUsers,nMovies,d)
        logJointLikelihoodObj = np.append(logJointLikelihoodObj,logJL)
    
    rsme = calculateRSME(PredictOnTestSet(ratingsTest,U,V),ratingsTest.rating)
    return({"U":U,"V":V,"logJLObj":logJointLikelihoodObj,"rsme":rsme})

# Make 10 runs and plot
tableVal = pd.DataFrame(columns=['run','logJLObj','RSME'])
V_dict = {}
for run in range(1,11):
    results=MAPMatrixCompletion(d,sigmasq,lambd,nMovies,nUsers,ratingsIndicator,ratingsMatrix,ratingsTest)
    V_dict["run"+str(run)]=results["V"]
    tableVal = tableVal.append({'run':run, 'logJLObj':results["logJLObj"][-1], 
                                'RSME':results['rsme']},ignore_index=True)
    plt.plot(np.arange(2,101),results["logJLObj"][1:],label="run "+str(run))
plt.title("Log Joint Likelihood")
plt.legend(loc='best')
plt.savefig("Problem2aLL.png")

tableVal=tableVal.sort_values(by='logJLObj',ascending=False)
tableVal[["run","logJLObj","RSME"]].to_csv("ObjectiveTable.csv")

# Get the closest 10 movies by Euclidean distance
movieTitles = pd.read_table("./hw4-data/movies.txt",header=None).values
V = V_dict["run5"]
queryMovies = ["Star Wars (1977)", "My Fair Lady (1964)", "GoodFellas (1990)"]

def getTenClosestMovies(queryMovie,V,movieTitles):
    
    movieV = V[:,np.where(movieTitles==queryMovie)[0]]    
    distances = np.sqrt(np.diag(np.dot((V - movieV).T,(V - movieV))))
    closestMovies = movieTitles[np.argsort(distances)[:11]]
    closestDistances = distances[np.argsort(distances)[:11]]
    
    #Create pandas dataframe
    movieResult = pd.DataFrame({"Movies":closestMovies.flatten(),
                                "Distance":closestDistances.flatten()})
    print(movieResult)
    #movieResult.to_csv("Problem2b_"+queryMovie+".csv")
    return(movieResult)

for m in queryMovies:
    getTenClosestMovies(m,V,movieTitles)
    


