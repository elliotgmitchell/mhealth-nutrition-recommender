#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 20:47:45 2018

@author: jeremylew, elliotgmitchell
"""

import pandas as pd
from Recommender import UserRatings, collabFilteringModel, getClosestFoods, getUserFavFoods;

user_food_table = pd.read_csv("Dataset.csv")
lunch = UserRatings(user_food_table,"lunch")

# Train the model
model = collabFilteringModel(d=8, sigmasq=0.6, lambd=1, nUsers=lunch.nUsers, nFoods=lunch.nFoods)
model.fit(lunch.ratings_train.values,seed_u=3,seed_v=5) #set the seed to get reproducible results

# Get model results
getClosestFoods("chicken and dumpling soup",lunch,model)
getUserFavFoods(445, lunch, model)
