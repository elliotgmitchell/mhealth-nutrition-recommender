import requests
import json
# from nutritionix import Nutritionix

app_id = ""
api_key = ""

send_me = {"appId":app_id,
        "appKey":api_key,
        # "query":"Cookies"
        "query":"Frap*",
        # "queries":{
        #     "item_name":"Kids Fries",
        #     "brand_name":"McDonalds"
        # }
        "filters":{
            "item_type":1
            }
        }

fancy_search = {
    "appId": app_id,
    "appKey": api_key,
    "fields": [
        "item_name",
        "brand_name",
        "nf_calories",
        "nf_sodium",
        "item_type",
        "nf_servings_per_container",
        "nf_serving_size_qty",
        "nf_serving_size_unit",
        "nf_serving_weight_grams",
        "nf_total_carbohydrate",
        "nf_dietary_fiber",
        "nf_sugars",
        "nf_protein"
    ],
    "offset": 0,
    "limit": 5,
    "sort": {
        "field": "nf_calories",
        "order": "desc"
    },
    # "min_score": 0.5,
    # "queries": {
    #   "brand_name":"Starbucks",
    #   "item_name":"grande AND latte"
    # },
    "query":"Grande AND Latte",
    "filters": {
        "not": {
            "item_type": 2
        }#,
    # "nf_calories": {
    #   "from": 0,
    #   "to": 400
    # },
    # "nf_total_carbohydrate": {
    #   "from": 2,
    #   "to": 30
    # }
    }
}

grande_latte = {
  "appId": app_id,
  "appKey": api_key,
  "fields": [
    "item_name",
    "brand_name",
    "nf_calories",
    "item_type",
    "nf_servings_per_container",
    "nf_serving_size_qty",
    "nf_serving_size_unit",
    "nf_serving_weight_grams",
    "nf_total_carbohydrate",
  ],
  "query":"Starbucks skim grande latte"
}

response = requests.post('https://api.nutritionix.com/v1_1/search',
                    headers = {'content-type': 'application/json'},
                    data = json.dumps(grande_latte))

# Version 2?
response = requests.post('https://trackapi.nutritionix.com/v2/natural/nutrients',
                         headers = {
                           "content-type": "application/json",
                           "x-app-id": app_id,
                           "x-app-key": api_key,
                           "x-remote-user-id": "0"
                         },
                         data = json.dumps({
                           "query":"butternut squash soup",
                         }))

print response.url

print json.dumps(response.json(),indent=2)

#
# {
#   "appId":app_id,
#   "appKey":api_key,
#   "fields":["item_name","brand_name","upc"],
#   "sort":{
#     "field":"_score",
#     "order":"desc"
#   },
#   "filters":{
#     "item_type":2
#   }
# }
