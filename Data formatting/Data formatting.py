#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:54:34 2018

@author: jeremylew
"""

import requests
import json
import pandas as pd
# from nutritionix import Nutritionix

app_id = "c0d12ddd"
api_key = "f7e2c9f01df5c6372c4bc9290a2abe8d"

# Version 2

def getFoodItems(user_id, meal_id, kind, query, user_foodID_table):
    response = requests.post('https://trackapi.nutritionix.com/v2/natural/nutrients',
                             headers = {
                               "content-type": "application/json",
                               "x-app-id": app_id,
                               "x-app-key": api_key
                             },
                             data = json.dumps({
                               "query":query,
                             }))
    foods_list=json.loads(response.text)["foods"]
    for food_dict in foods_list:
        dataToBeInserted = {"user_id": user_id, 
                            "meal_id": meal_id,
                            "food": food_dict["food_name"],
                            "calories": food_dict["nf_calories"],
                            "kind": kind}
        user_foodID_table = user_foodID_table.append(dataToBeInserted, ignore_index=True)
    return user_foodID_table    

def createUserFoodTable(platanoData, user_foodID_table):
    for index, row in platanoData.iterrows():
        food_query = row["title"] + " " + row["ingredients"]
        user_foodID_table = getFoodItems(row["user_id"],row["meal_id"],row["kind"],food_query, user_foodID_table)
    return user_foodID_table
    

platano = pd.read_csv("platano subset.csv")
user_foodID_table = pd.DataFrame(columns=["user_id","meal_id","food","calories","kind"]) #Creates empty data frame
user_foodID_table = createUserFoodTable(platano,user_foodID_table) # Generate a user-food table

