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


class UserRatings():
    
    def __init__(self,user_food_table,meal_kind=None):
        
        # =============================================================================
        # Data preprocessing
        # =============================================================================
        
        # Subset by meal
        if meal_kind==None:
            meal_table = user_food_table.iloc[:,1:]
        else:
            meal_table = user_food_table[user_food_table["kind"]==meal_kind].iloc[:,1:]
        
        # Give each food a foodID
        food_id_table = pd.DataFrame({"food":np.unique(meal_table.food),
                                      "foodID":np.arange(0,len(np.unique(meal_table.food)))}) #the index in this list is used to determine the food ID 
        
        # Get count of meals logged and foods taken
        food_count = meal_table[["user_id","food"]].groupby(["user_id","food"])["food"].count().reset_index(name="food count")
        food_count = food_count.merge(food_id_table,how="left",on="food")
        meals_logged_count = meal_table[["user_id","meal_id"]].groupby("user_id")["meal_id"].nunique().reset_index(name="count of meals logged")
        
        # Calculate the ratings
        ratings = food_count.merge(meals_logged_count,how="left",on="user_id")
        ratings["ratings"] = ratings["food count"]/ratings["count of meals logged"]  #calculate ratings
        
        # Remove the mean from the ratings 
        ratings["ratings"] = ratings.ratings - np.mean(ratings.ratings)

        # Give each user an index to determine their position in the matrix
        user_id_index_table = pd.DataFrame({"user_id":np.unique(ratings.user_id),
                                            "user_id_index":np.arange(0,len(np.unique(ratings.user_id)))})
        ratings = ratings.merge(user_id_index_table,how="left",on="user_id")    
    
        # Ratings attributes
        self.ratings = ratings
        self.food_id_table = food_id_table
        self.user_id_index_table = user_id_index_table
        self.nUsers = len(self.user_id_index_table)
        self.nFoods = len(self.food_id_table)
                
        # =============================================================================
        # Train-test split
        # =============================================================================
        
        ratings_train, ratings_test = train_test_split(self.ratings,test_size=0.2,random_state=30)
        
        # Train/test attributes
        self.ratings_train = ratings_train
        self.ratings_test = ratings_test
              
    # =============================================================================
    # Populate the matrix 
    # =============================================================================

    def populateMatrix(self):
        
        # Initialise matrix 
        self.ratingsMatrix = np.empty((self.nUsers,self.nFoods))*np.nan 
        self.ratingsIndicator = np.zeros((self.nUsers,self.nFoods))
        
        # Populate the ratings matrix and ratings indicator matrix with training data
        for index,user_id_index,foodID,ratings in self.ratings_train[["user_id_index","foodID","ratings"]].itertuples():
            self.ratingsIndicator[user_id_index,foodID] = 1
            self.ratingsMatrix[user_id_index,foodID] = ratings


user_food_table = pd.read_csv("Dataset.csv")
lunch = UserRatings(user_food_table,"lunch")
lunch.populateMatrix()
  
# =============================================================================
# Matrix factorization 
# =============================================================================

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

class collabFilteringModel(BaseEstimator):
    user_food_table = pd.read_csv("Dataset.csv")
    
    def __init__(self,d=2,sigmasq=0.25,lambd=1,UserRatings=UserRatings(user_food_table)):
        
        # Parameters
        self.sigmasq = sigmasq
        self.d = d
        self.lambd = lambd 
        self.UserRatings = UserRatings
        
#    def _logJointLikelihood(self,UserRatings): 
#        L = - np.sum([(1/(2*self.sigmasq))*
#                   (UserRatings.ratingsMatrix[row,column]-np.dot(self.U[row,:],V[:,column]))**2 
#                    for (row,column) in list(zip(*np.where(UserRatings.ratingsIndicator==1)))]) \
#                    - (self.lambd/2)*np.matrix.trace(np.dot(U,U.T)) \
#                    - (self.lambd/2)*np.matrix.trace(np.dot(V.T,V)) \
#                    - len(np.where(UserRatings.ratingsIndicator==1)[0]) * np.log(np.sqrt((2*np.pi*self.sigmasq))) \
#                    - UserRatings.nUsers * np.log((2*np.pi/self.lambd)**(self.d/2)) \
#                    - UserRatings.nFoods * np.log((2*np.pi/self.lambd)**(self.d/2))
#        return(L)
    
    def fit(self,X,y=None):
        
        #initialize all N1 Ui and N2 vj
        V = np.random.multivariate_normal(mean=np.zeros(self.d),cov=(1/self.lambd)*np.eye(self.d),size=self.UserRatings.nFoods).T
        U = np.random.multivariate_normal(mean=np.zeros(self.d),cov=(1/self.lambd)*np.eye(self.d),size=self.UserRatings.nUsers)
#        logJointLikelihoodObj = np.array([])
        
        for t in range(0,100):
            
            #Update user location
            for i in range(0,UserRatings.nUsers):
                ratedObjIndexes = np.where(UserRatings.ratingsIndicator[i,:]==1)[0]
                Vrated = V[:,ratedObjIndexes]
                Mrated = UserRatings.ratingsMatrix[i,ratedObjIndexes]
                Vsum = np.dot(Vrated,Vrated.T)
                U[i,:]=np.dot(np.linalg.inv(self.lambd*self.sigmasq*np.eye(self.d) + Vsum), np.dot(Vrated,Mrated.T))
            
            #Update object location
            for j in range(0,UserRatings.nFoods):
                ratedUserIndexes = np.where(UserRatings.ratingsIndicator[:,j]==1)[0]
                Urated = U[ratedUserIndexes,:]
                Mrated = UserRatings.ratingsMatrix[ratedUserIndexes,j]           
                Usum = np.dot(Urated.T,Urated)            
                V[:,j] = np.dot(np.linalg.inv(self.lambd*self.sigmasq*np.eye(self.d) + Usum), np.dot(Urated.T,Mrated))
        
#            logJL = self._logJointLikelihood(UserRatings)
#            logJointLikelihoodObj = np.append(logJointLikelihoodObj,logJL)
        
#        self.logJointLikelihood = logJointLikelihoodObj
        self.U_ = U
        self.V_ = V
    
        return self
    
    def predict(self,UserRatings):
        predictedRatings = np.array([])
        for index, user_id_index, foodID, ratings in UserRatings.ratings_test[["user_id_index","foodID","ratings"]].itertuples():
            predictedRatings = np.append(predictedRatings,
                                         np.dot(self.U_[user_id_index,:], 
                                                self.V_[:,foodID]))
        return predictedRatings

    def score(self,UserRatings):
        # Calculates the RSME
        predicted = self.predict(UserRatings)
        if len(predicted) != len(UserRatings.ratings_test): 
            raise Exception("Vector lengths not equal")
        else: 
            rsme = np.sqrt((1/len(UserRatings.ratings_test))*np.sum((UserRatings.ratings_test["ratings"]-predicted)**2))
            return -rsme #sklearn treats bigger as better    
            
params = {"d": np.arange(2,18,2),
          "sigmasq":[0.05,0.10,0.25,0.35,0.45,0.5,0.6,0.8],
          "lambd":[1,2,3],
          "UserRatings":lunch}

gridSearch = GridSearchCV(collabFilteringModel(),param_grid=params, cv=10)

        
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