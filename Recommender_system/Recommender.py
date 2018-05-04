#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:47:15 2018

@author: jeremylew, elliotgmitchell
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules


class UserRatings():

    def __init__(self,user_food_table,meal_kind=None, trim_low_food_count=1, trim_threshold=1):

        # =============================================================================
        # Data preprocessing
        # =============================================================================

        # Subset by meal
        if meal_kind==None:
            self.meal_table = user_food_table.iloc[:,1:]
        else:
            self.meal_table = user_food_table[user_food_table["kind"]==meal_kind].iloc[:,1:]

        # Filter out calorie free foods like 'bottle of water'
        self.meal_table = self.meal_table[self.meal_table.calories > 5]

        # Remove low frequency foods in meal table
        if trim_low_food_count == 1:
            filtered = self.meal_table.groupby(["food"])["food"].filter(lambda x: len(x) > trim_threshold)
            self.meal_table = self.meal_table[self.meal_table["food"].isin(filtered)]

        # Give each food a foodID
        food_id_table = self.meal_table[["food","carbs","protein","fat","fiber","calories"]].copy().drop_duplicates(subset="food").reset_index(drop=True)
        food_id_table["foodID"] = np.arange(0,len(np.unique(self.meal_table.food)))

        # Get count of meals logged and foods taken per user
        food_count = self.meal_table[["user_id","food"]].groupby(["user_id","food"])["food"].count().reset_index(name="food count")
        food_count = food_count.merge(food_id_table,how="left",on="food")
        meals_logged_count = self.meal_table[["user_id","meal_id"]].groupby("user_id")["meal_id"].nunique().reset_index(name="count of meals logged")

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

    def calculateAssociationRules(self, min_support=0.01):

        # Format foods for apriori algorithm
        food_item_table = self.meal_table.groupby(["meal_id","food"])["user_id"]\
                            .count().unstack().reset_index().fillna(0).set_index("meal_id")
        food_sets = food_item_table.applymap(encode_units)

        frequent_foodsets = apriori(food_sets, min_support=min_support, use_colnames=True)
        self.rules = association_rules(frequent_foodsets)

        return self.rules

    def mealSummary(self, user_id):
        """
        Return a summary of meals eaten for this user
        """
        user_meals = self.meal_table[(self.meal_table.user_id == user_id)]
        user_food_list = user_meals.food.unique()
        user_meal_summary = pd.DataFrame({'list of foods': user_meals.groupby("meal_id")['food'].apply(', '.join)})
        return user_food_list, user_meal_summary


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

if __name__ == '__main__':
    user_food_table = pd.read_csv("Dataset.csv")
    lunch = UserRatings(user_food_table,"lunch",trim_low_food_count=1, trim_threshold=1)


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

    def fit(self,X,y=None,**kwargs):
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

        #initialize all N1 Ui and N2 Vj
        if kwargs.get("seed_v"): np.random.seed(kwargs.get("seed_v"))
        V = np.random.multivariate_normal(mean=np.zeros(self.d),cov=(1/self.lambd)*np.eye(self.d),size=self.nFoods).T

        if kwargs.get("seed_u"): np.random.seed(kwargs.get("seed_u"))
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
if __name__ == '__main__':
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


if __name__ == '__main__':
    V_dict, tableVal = makePredictions(lunch)

# =============================================================================
# Make predictions using model
# =============================================================================

# Get the closest Foods by Euclidean distance
def getClosestFoods(queryFood, UserRatings, model):
    foodIndex = np.where(UserRatings.food_id_table==queryFood)[0]
    foodV = model.V_[:,foodIndex]
    distances = np.sqrt(np.diag(np.dot((model.V_ - foodV).T,(model.V_ - foodV))))

    #Create pandas dataframe
    foodResult = UserRatings.food_id_table.copy()
    foodResult["Distance"] = distances
    foodResult=foodResult.drop(foodIndex[0])
    foodResult = foodResult.sort_values(by=["Distance"])

    return foodResult

def getUserFavFoods(user_id, UserRatings, model):
    user_id_index = np.where(UserRatings.user_id_index_table["user_id"]==user_id)[0][0]
    userRatings = np.dot(model.U_[user_id_index,:],model.V_).flatten()

    # Favourite foods according to rating scores
    fav = UserRatings.food_id_table.copy()
    fav["Inferred ratings"] = userRatings
    fav = fav.merge(UserRatings.ratings_train[UserRatings.ratings_train["user_id_index"]==user_id_index][["foodID","ratings"]],how="left",on="foodID")
    fav = fav.merge(UserRatings.ratings_test[UserRatings.ratings_test["user_id_index"]==user_id_index][["foodID","ratings"]],how="left",on="foodID",suffixes=("_train","_test"))
    fav = fav.sort_values(by=["Inferred ratings"],ascending=False)

    return fav


def getMealRec(user_id, user_ratings, model, seed_food, nutrition_constraints=None):
    """
    Make recommendations for a user, based on a given seed food item
    For example, given the seed_food 'chicken', return a dataframe of possible
        food items to add to the seed food item, such that the combined meal is
        within the given nutrition_constraints

    TODO: Implement nutrition_constraints variable. For now, hard coded constraints.
    """

    # Get association rules and extract candidate food sets
    try:
        rules = user_ratings.rules
    except AttributeError:
        rules = user_ratings.calculateAssociationRules(0.001)
    candidates = rules[rules.antecedants.apply(lambda x: seed_food in x)].consequents.unique()

    # Get user fav foods from collabFilteringModel
    fav = getUserFavFoods(user_id, user_ratings, model)

    # df of nutrition information
    seed_food_nutrition = user_ratings.food_id_table[user_ratings.food_id_table.food == seed_food]

    meal_recommendations = pd.DataFrame()

    # For each candidate, update the output df with a combination that meets the constraints
    for candidate in candidates:
        # print("trying {}".format(candidate))
        meal_recommendations = check_nutrition_constraints(meal_recommendations, seed_food_nutrition, candidate, fav, nutrition_constraints, user_ratings)

    # sort by the score from the recommender algorithm
    if not meal_recommendations.empty:
        meal_recommendations.sort_values(by="inferred_score", ascending=False, inplace=True)

    return meal_recommendations


def check_nutrition_constraints(meal_recommendations, seed_food_nutrition, candidate, fav, nutrition_constraints, user_ratings):
    """
    Helper function to see if some combination of candidate foods and seed food
    can be used to meet nutrition_constraints
    For example, adding 2 servings of carrots to 1 servings of chicken

    TODO: nutrition_constraints is hard-coded for now.
    """

    # Nutrition information for candidate_food_items
    candidate_food_items = user_ratings.food_id_table[user_ratings.food_id_table.food.apply(lambda x: x in candidate)]

    serving_options = [0.5, 1, 2]

    for i in serving_options: # n servings of seed food
        for j in serving_options: # n servings of candidates

            # Calculate carbs and calories
            carbs = (seed_food_nutrition.carbs.iloc[0] * i) + (candidate_food_items.carbs.sum() * j)
            calories = (seed_food_nutrition.calories.iloc[0] * i) + (candidate_food_items.calories.sum() * j)

            if ((carbs > 30) and (carbs < 95)) and ((calories > 300) and (calories < 600)):
                # Look up preference from `fav`
                score = fav[fav.food.apply(lambda x: x in candidate)]["Inferred ratings"].mean()

                # Add to output dataframe
                temp_df = pd.DataFrame({
                    "candidates": candidate,
                    "inferred_score": score,
                    "seed_servings": i,
                    "candidate_servings": j
                }).drop_duplicates()

                meal_recommendations = meal_recommendations.append(temp_df)
                return meal_recommendations
    # print("could not find a combination within constraints")
    return meal_recommendations


def format_meal_rec(suggestion, seed_food):
    meal_rec = "To go with {candidate_servings:g} {candidate_servings_sing_or_plural} \
of {seed_food}, try {seed_servings:g} {seed_servings_sing_or_plural} of {candidates}".format(
        candidate_servings = suggestion.candidate_servings,
        candidate_servings_sing_or_plural = servings_sing_or_plural(suggestion.candidate_servings),
        seed_food = seed_food,
        seed_servings = suggestion.seed_servings,
        seed_servings_sing_or_plural = servings_sing_or_plural(suggestion.seed_servings),
        candidates = format_candidates(suggestion.candidates)
)
    return meal_rec


def servings_sing_or_plural(n_servings):
    if n_servings > 1:
        return "servings"
    return "serving"


def format_candidates(candidates):
    candidates = list(candidates)
    if len(candidates) == 1:
        return candidates[0]
    candidates_str = ', '.join(candidates[0:-1])
    candidates_str += ' and {}'.format(candidates[-1])
    return candidates_str


if __name__ == '__main__':
    model = collabFilteringModel(d=8, sigmasq=0.6, lambd=1, nUsers=lunch.nUsers, nFoods=lunch.nFoods)
    model.fit(lunch.ratings_train.values)
    print(getClosestFoods("chicken and dumpling soup",lunch,model))
    print(getUserFavFoods(445, lunch, model))
