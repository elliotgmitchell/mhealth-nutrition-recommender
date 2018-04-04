import os
import sys
import string

from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd

import matplotlib.pyplot as plt

PATH_TO_DATA =  "./platano_dump.csv"


# Read dataframe from CSV
food_data = pd.read_csv(PATH_TO_DATA, encoding='utf-8')

# Remove NA titles and descriptions
food_data = food_data.dropna(subset=["ingredients", "title"])

# Set of stopwords to remove from meal titles and descriptions
stop = stopwords.words('english') + list(string.punctuation)

def tokenize_food_items(food_item):
    """
    Quick/crude function to tokenize food titles and descriptions
    """
    food_items_processed = word_tokenize(food_item)
    # print(food_items_processed)
    food_items_processed = [t.strip() for t in food_items_processed]
    food_items_processed = [t for t in food_items_processed if t not in stop]
    food_items_processed = [t for t in food_items_processed if (not t[0].isdigit())]
    food_items_processed = [t.lower() for t in food_items_processed]
    # print(food_items_processed)
    return " ".join(food_items_processed)




def item_freq():
    """
    Print some stats about how frequently food item tokens appear in the dataset
    """

    food_data["ingredients_tokenized"] = food_data.ingredients.apply(tokenize_food_items)

    # print(food_data[["user_id", "ingredients", "ingredients_tokenized"]])

    # Aggregate by user_id
    user_food_counts = food_data.groupby("user_id")["ingredients_tokenized"].apply(" ".join)

    # print(user_food_counts)

    cv = CountVectorizer()
    counts = cv.fit_transform(user_food_counts)

    user_matrix = pd.DataFrame(counts.toarray(), columns=cv.get_feature_names())
    # print(user_matrix)
    # print(user_matrix.shape)

    # How many times is each food item logged in the dataset?
    item_counts = user_matrix.sum(axis=0)
    # print(item_counts)
    # print(item_counts.sort_values())
    print("Number of food items logged 25 times or more:")
    print(len(item_counts[item_counts > 25]))

    # How many food items are in more than one log?
    user_matrix_binary = user_matrix.applymap(lambda x: x > 0)
    # print(user_matrix_binary)
    binary_counts = user_matrix_binary.sum(axis=0)
    # print(binary_counts)
    print("Number of food items that appear in more than one user log:")
    print(len(binary_counts[binary_counts > 1]))



def field_lenths():
    """
    Print some stats about the average lengths of meal titles and descriptions
    """
    # Average ingredient length
    food_data["ingredients_length"] = food_data.ingredients.apply(lambda x: len([t for t in word_tokenize(x) if t not in string.punctuation]))
    # print(food_data[["meal_id", "ingredients", "ingredients_length"]])

    # Average description Lengh
    food_data["title_length"] = food_data.title.apply(lambda x: len([t for t in word_tokenize(x) if t not in string.punctuation]))
    # print(food_data[["meal_id", "title", "title_length"]])

    print("Title average length:")
    print("\tMedian: {}".format(food_data.title_length.median()))
    print("\tMean: {}".format(food_data.title_length.mean()))
    print("Description average length:")
    print("\tMedian: {}".format(food_data.ingredients_length.median()))
    print("\tMean: {}".format(food_data.ingredients_length.mean()))

    # plt.hist(food_data.ingredients_length, bins=50, density=True)
    # plt.show()


# for _, food_item in food_data.ingredients.iteritems():
#     if pd.isnull(food_item):
#         continue
#     print(food_item)
#     food_items_processed = word_tokenize(food_item)
#     print(food_items_processed)
#     food_items_processed = [t.lower() for t in food_items_processed if t not in stop]
#     print(food_items_processed)

if __name__ == '__main__':
    item_freq()
    field_lenths()
