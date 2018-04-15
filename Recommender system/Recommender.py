#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:47:15 2018

@author: jeremylew, elliotgmitchell
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# =============================================================================
# Data preprocessing
# =============================================================================

user_food_table = pd.read_csv("Dataset.csv")

def generateRatingsTable(user_food_table,meal_kind):
    
    # Subset by meal
    meal_table = user_food_table[user_food_table["kind"]==meal_kind].iloc[:,1:]
    
    # Give each food a foodID
    food_id_table = pd.DataFrame({"food":np.unique(meal_table.food),"foodID":np.arange(0,len(np.unique(meal_table.food)))}) #the index in this list is used to determine the food ID 
    
    # Get count of meals logged and foods taken
    food_count = meal_table[["user_id","food"]].groupby(["user_id","food"])["food"].count().reset_index(name="food count")
    food_count = food_count.merge(food_id_table,how="left",on="food")
    meals_logged_count = meal_table[["user_id","meal_id"]].groupby("user_id")["meal_id"].nunique().reset_index(name="count of meals logged")
    
    # Calculate the ratings
    ratings = food_count.merge(meals_logged_count,how="left",on="user_id")
    ratings["ratings"] = ratings["food count"]/ratings["count of meals logged"]  #calculate ratings
    
    # Remove the mean from the ratings 
    ratings["ratings"] = ratings.ratings - np.mean(ratings.ratings)
    
    return ratings,food_id_table

lunch_ratings, lunch_food_id = generateRatingsTable(user_food_table,"lunch")


# =============================================================================
# Train-test split
# =============================================================================

X_train, X_test = train_test_split(lunch_ratings,test_size=0.2,random_state=30)

# =============================================================================
# Populate the matrix 
# =============================================================================

def populateMatrix(ratings,food_id_table):
    nUsers=len(np.unique(ratings.user_id))
    nFoods=len(np.unique(ratings.foodID))
    
    # Initialise matrix 
    ratingsMatrix = np.empty((nUsers,nFoods))*np.nan 
    ratingsIndicator = np.zeros((nUsers,nFoods))
    
    # Populate the ratings matrix and ratings indicator matrix 
    for index,row in ratings[["user_id","foodID","ratings"]]:
        ratingsIndicator[[row,"user_id"]-1,ratingsCSV.loc[row,"movie_id"]-1] = 1
        ratingsMatrix[ratingsCSV.loc[row,"user_id"]-1,ratingsCSV.loc[row,"movie_id"]-1] = ratingsCSV.loc[row,"rating"]
    

populateMatrix(lunch_ratings)

  
# =============================================================================
# Matrix factorization 
# =============================================================================

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