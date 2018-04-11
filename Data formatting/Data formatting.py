#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:54:34 2018

@author: jeremylew, elliotgmitchell
"""

import os
import requests
import json
import pandas as pd

app_id = ""
api_key = ""

# file path for raw data dump
# data_path = "/Users/jeremylew/Dropbox (Personal)/mHealth Project/Data/data_dump_2.csv" # Jeremy's path
data_path = os.path.expanduser("~/Dropbox/Columbia/Spring 2018/Data Science for Mobile Health/mHealth Project/Data/data_dump_2.csv") # Elliot's path

# file path for processed output
output_path = "./egm2143_processed_data.csv"

# Version 2

def getFoodItems(user_id, meal_id, kind, query, user_foodID_table):
    try:
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
                                "carbs": food_dict["nf_total_carbohydrate"],
                                "fat": food_dict["nf_total_fat"],
                                "protein": food_dict["nf_protein"],
                                "fiber": food_dict["nf_dietary_fiber"],
                                "calories": food_dict["nf_calories"],
                                "kind": kind}
            user_foodID_table = user_foodID_table.append(dataToBeInserted, ignore_index=True)
        return user_foodID_table

    except KeyError:
        print("Invalid query:",query)
        return user_foodID_table


def createUserFoodTable(platanoData, user_foodID_table):
    for index,row in platanoData.iterrows():
        try:
            food_query = str(row["title"]) + " " + str(row["ingredients"])
            user_foodID_table = getFoodItems(row["user_id"],row["meal_id"],row["kind"],food_query, user_foodID_table)
        except:
            print("Invalid query:", index, row["title"], row["ingredients"])
            continue

    return user_foodID_table


# Read in Platano data
platano = pd.read_csv(data_path)
platano[["title","ingredients"]] = platano[["title","ingredients"]].fillna("")

# Subset due to API limit:
platano_subset = platano.loc[2:4,:]

# Get user-food table
try:
    user_foodID_table = pd.read_csv(output_path)
    print("found existing output file")
except FileNotFoundError as e:
    user_foodID_table = pd.DataFrame(columns=["user_id","meal_id","food","carbs","protein","fat","fiber","calories","kind"]) #Creates empty data frame
    print("starting output from scratch")

user_foodID_table = createUserFoodTable(platano_subset,user_foodID_table) # Generate a user-food table

# Export file
user_foodID_table.to_csv(output_path, index=False)
