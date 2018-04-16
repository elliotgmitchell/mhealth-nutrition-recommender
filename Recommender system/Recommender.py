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
        self.ratings = ratings[["user_id_index","foodID","ratings"]]
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
              
    
user_food_table = pd.read_csv("Dataset.csv")
lunch = UserRatings(user_food_table,"lunch")

  
# =============================================================================
# Matrix factorization 
# =============================================================================

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

class collabFilteringModel(BaseEstimator):
    
    def __init__(self,d=2,sigmasq=0.25,lambd=1,nUsers=10,nFoods=10):
        
        # Parameters
        self.sigmasq = sigmasq
        self.d = d
        self.lambd = lambd 
        self.nUsers = nUsers
        self.nFoods = nFoods
        
    def _logJointLikelihood(self,ratingsMatrix,ratingsIndicator,U,V): 
        L = - np.sum([(1/(2*self.sigmasq))*
                   (ratingsMatrix[row,column]-np.dot(U[row,:],V[:,column]))**2 
                    for (row,column) in list(zip(*np.where(ratingsIndicator==1)))]) \
                    - (self.lambd/2)*np.matrix.trace(np.dot(U,U.T)) \
                    - (self.lambd/2)*np.matrix.trace(np.dot(V.T,V)) \
                    - len(np.where(ratingsIndicator==1)[0]) * np.log(np.sqrt((2*np.pi*self.sigmasq))) \
                    - self.nUsers * np.log((2*np.pi/self.lambd)**(self.d/2)) \
                    - self.nFoods * np.log((2*np.pi/self.lambd)**(self.d/2))
        return(L)
    
    def fit(self,X,y=None):
        # X is UserRatings.ratings_train.values
        
        # Initialise matrix 
        ratingsMatrix = np.empty((self.nUsers,self.nFoods))
        ratingsMatrix[:] = np.nan 
        ratingsIndicator = np.zeros((self.nUsers,self.nFoods))
        
        # Populate the ratings matrix and ratings indicator matrix with training data
        user_food_index = X[:,0:2].astype(int)
        ratingsValue = X[:,2]
        for i in range(0,len(user_food_index)):
            ratingsIndicator[user_food_index[i,0],user_food_index[i,1]] = 1
            ratingsMatrix[user_food_index[i,0],user_food_index[i,1]] = ratingsValue[i]
        
        #initialize all N1 Ui and N2 vj
        V = np.random.multivariate_normal(mean=np.zeros(self.d),cov=(1/self.lambd)*np.eye(self.d),size=self.nFoods).T
        U = np.random.multivariate_normal(mean=np.zeros(self.d),cov=(1/self.lambd)*np.eye(self.d),size=self.nUsers)
        logJointLikelihoodObj = np.array([])
            
        for t in range(0,100):
            
            #Update user location
            for i in range(0,self.nUsers):
                ratedObjIndexes = np.where(ratingsIndicator[i,:]==1)[0]
                Vrated = V[:,ratedObjIndexes]
                Mrated = ratingsMatrix[i,ratedObjIndexes]
                Vsum = np.dot(Vrated,Vrated.T)
                U[i,:]=np.dot(np.linalg.inv(self.lambd*self.sigmasq*np.eye(self.d) + Vsum), np.dot(Vrated,Mrated.T))
            
            #Update object location
            for j in range(0,self.nFoods):
                ratedUserIndexes = np.where(ratingsIndicator[:,j]==1)[0]
                Urated = U[ratedUserIndexes,:]
                Mrated = ratingsMatrix[ratedUserIndexes,j]           
                Usum = np.dot(Urated.T,Urated)            
                V[:,j] = np.dot(np.linalg.inv(self.lambd*self.sigmasq*np.eye(self.d) + Usum), np.dot(Urated.T,Mrated))
        
            logJL = self._logJointLikelihood(ratingsMatrix,ratingsIndicator,U,V)
            logJointLikelihoodObj = np.append(logJointLikelihoodObj,logJL)
        
        self.logJointLikelihood = logJointLikelihoodObj
        self.U_ = U
        self.V_ = V
    
        return self
    
    def predict(self,X,y=None):
        # X is UserRatings.ratings_test.values
        user_food_index = X[:,0:2].astype(int)
        predictedRatings = np.array([])
        
        for i in range(0,len(user_food_index)):
            predictedRatings = np.append(predictedRatings,
                                         np.dot(self.U_[user_food_index[i,0],:], 
                                                self.V_[:,user_food_index[i,1]]))
        return predictedRatings

    def score(self,X,y=None):
        # X is UserRatings.ratings_train.values or UserRatings.ratings_test.values
        y = X[:,2]
        
        # Calculates the RSME
        predicted = self.predict(X,y)
        if len(predicted) != len(y): 
            raise Exception("Vector lengths not equal")
        else: 
            rsme = np.sqrt((1/len(y))*np.sum((y-predicted)**2))
            return -rsme #sklearn treats bigger as better    
 
# =============================================================================
# Select best model parameters using cross validation
# =============================================================================
           
params = {"d":np.arange(2,14,2),
          "sigmasq":[0.05,0.10,0.25,0.35,0.45,0.5,0.6,0.8],
          "lambd":[1,2,3]}

gridSearch = GridSearchCV(collabFilteringModel(nUsers=lunch.nUsers,nFoods=lunch.nFoods), param_grid=params, cv=5)
gridSearch.fit(lunch.ratings_train.values)

print(gridSearch.best_params_)
print(gridSearch.best_score_)

#print(gridSearch.best_params_)
#{'d': 8, 'lambd': 1, 'sigmasq': 0.6}

#print(gridSearch.best_score_)
#-0.338966135715

#print(gridSearch.cv_results_,file=open("./CV results.txt","w"))
#print(gridSearch.grid_scores_,file=open("./grid scores.txt","w"))

# =============================================================================
# Check out the variability in log joint likelihood (objective function) between model runs using best parameters  
# =============================================================================
        
# Make 10 runs and plot

def makePredictions(UserRatings):
    X_train = UserRatings.ratings_train.values
    X_test = UserRatings.ratings_test.values
    
    tableVal = pd.DataFrame(columns=['run','logJLObj','RSME'])
    V_dict = {}
    
    for run in range(1,11):
        model = collabFilteringModel(d=8, sigmasq=0.6, lambd=1, nUsers=UserRatings.nUsers, nFoods=UserRatings.nFoods)
        model.fit(X_train)
        V_dict["run"+str(run)]=model.V_
        tableVal = tableVal.append({'run':run, 'logJLObj':model.logJointLikelihood[-1], 
                                    'RSME':-model.score(X_test,)},ignore_index=True)
        plt.plot(np.arange(2,101),model.logJointLikelihood[1:],label="run "+str(run))

    plt.title("Log Joint Likelihood")
    plt.legend(loc='best')
    plt.savefig("./Objective Function.png")

    tableVal=tableVal.sort_values(by='logJLObj',ascending=False)
    tableVal[["run","logJLObj","RSME"]].to_csv("ObjectiveTable.csv")

    return V_dict, tableVal


V_dict, tableVal = makePredictions(lunch)

# =============================================================================
# Make predictions using model
# =============================================================================

model = collabFilteringModel(d=8, sigmasq=0.6, lambd=1, nUsers=lunch.nUsers, nFoods=lunch.nFoods)
model.fit(lunch.ratings_train.values)

# Get the closest 10 Foods by Euclidean distance
def getClosestFoods(queryFood, food_id_table, V):
    foodIndex = np.where(food_id_table==queryFood)[0]
    foodV = V[:,foodIndex]    
    distances = np.sqrt(np.diag(np.dot((V - foodV).T,(V - foodV))))
    
    closestFoods = food_id_table.loc[np.argsort(distances)]["food"]
    closestDistances = distances[np.argsort(distances)]
    
    #Create pandas dataframe
    foodResult = pd.DataFrame({"Closest Foods":closestFoods,
                                "Distance":closestDistances.flatten()})
    foodResult=foodResult.drop(foodIndex)
    
    return(foodResult)

getClosestFoods("chicken and dumpling soup",lunch.food_id_table,model.V_)

def getUserFavFoods(user_id, UserRatings, model):
    user_id_index = np.where(UserRatings.user_id_index_table["user_id"]==user_id)[0]
    userRatings = np.dot(model.U_[user_id_index,:],model.V_).flatten()
    
    # Favourite foods according to rating scores
    fav = pd.DataFrame({"Food":UserRatings.food_id_table["food"],"Scores":userRatings})
    fav = fav.sort_values(by=["Scores"],ascending=False)
    
    return fav

getUserFavFoods(445, lunch, model)



