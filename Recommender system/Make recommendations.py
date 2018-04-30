#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 20:47:45 2018

@author: jeremylew, elliotgmitchell
"""

import pandas as pd
from Recommender import UserRatings, collabFilteringModel, getClosestFoods, getUserFavFoods, getMealRec;

user_food_table = pd.read_csv("Dataset.csv")
lunch = UserRatings(user_food_table)

# Train the model
print("Training the model...")
model = collabFilteringModel(d=8, sigmasq=0.6, lambd=1, nUsers=lunch.nUsers, nFoods=lunch.nFoods)
model.fit(lunch.ratings_train.values,seed_u=3,seed_v=5) #set the seed to get reproducible results

# Get model results
print("Inspecting the model output...")
closest_foods = getClosestFoods("chicken and dumpling soup",lunch,model)
fav_foods = getUserFavFoods(445, lunch, model)

print(closest_foods)
print(fav_foods)

print("Calculating association rules...")
lunch.calculateAssociationRules(0.001)
print(lunch.rules)
print(lunch.rules[lunch.rules.antecedants.apply(lambda x: 'chicken' in x)])

user_id = 445
seed_food = 'chicken'
foo = getMealRec(user_id, user_ratings=lunch, model=model, seed_food=seed_food, nutrition_constraints="")
print("Suggestions to go with {}:".format(seed_food))
print(foo)
