# nutrition-recommender

A recommender system for healthy nutrition.

@authors: jeremylew, elliotgmitchell

# Jupyter Notebooks

These Jupyter notebooks contain examples of the complete recommendation system algorithm for three different user/meal type combinations:

    Evaluate Recommendations User 1665 Breakfast.ipynb
    Evaluate Recommendations User 445 Lunch.ipynb
    Evaluate Recommendations User 1821 Dinner.ipynb


# `/Data formatting/`
Contains the code used to preprocess and format the original Mealyzer data into a format for collaborative filtering and subsequent analysis.

Performs toakenization of free-text food descriptions using the Nutritionix natural language API.
https://www.nutritionix.com/natural-demo


# `/Recommender_system/`
The code and output for collaborative filtering, association rules, and the complete meal recommendation algorithm.

## `Recommender.py`
Includes the key classes and methods for running a collaborative filtering model and receiving user-specific food recommendations.

#### Classes

`UserRatings(user_food_table, meal_kind=None, trim_low_food_count=1, trim_threshold=1)`

Methods:

`calculateAssociationRules`

`mealSummary`

`collabFilteringModel(d=2,sigmasq=0.25,lambd=1,nUsers=10,nFoods=10)`

`fit`

`predict`

`score`

#### Functions

`getClosestFoods(queryFood, UserRatings, model)`

`getUserFavFoods(user_id, UserRatings, model)`

`getMealRec(user_id, user_ratings, model, seed_food, nutrition_constraints=None)`

## `Make recommendations.py`

Includes a sample script to run the complete recommendation system algorithm.
