import requests
import os
import json
import csv

from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env file

# export 'BEARER_TOKEN'='AAAAAAAAAAAAAAAAAAAAACHGqQEAAAAAH3l%2BlJ8AqwBjCKw83leN84FQJxw%3D2FdcSyni6gm9DmHteNclyIAqrozDnIDS3LWA5wn2KRLEMORp1i'
bearer_token = os.environ.get("BEARER_TOKEN")
# bearer_token = "AAAAAAAAAAAAAAAAAAAAACHGqQEAAAAAH3l%2BlJ8AqwBjCKw83leN84FQJxw%3D2FdcSyni6gm9DmHteNclyIAqrozDnIDS3LWA5wn2KRLEMORp1i"


def find_user_id(username):
    url = f"https://api.twitter.com/2/users/by/username/{username}"
    response = requests.request("GET", url, auth=bearer_oauth)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    json_response = response.json()
    return json_response['data']['id']

def create_url(username):
    user_id = find_user_id(username)
    return "https://api.twitter.com/2/users/{}/tweets".format(user_id)

def get_params(next_token=None):
    params = {
        "expansions": "referenced_tweets.id,author_id",
        "tweet.fields": "created_at,public_metrics,text", 
        "user.fields": "username",
        "max_results": 100
    }
    if next_token:
        params['pagination_token'] = next_token
    return params

def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2UserTweetsPython"
    return r

def connect_to_endpoint(url, params):
    response = requests.request("GET", url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()

# def main():
#     username = "WinstonPortal"  # change this to the username you're interested in
#     url = create_url(username)
#     params = get_params()
#     with open('tweets70.csv', 'w', newline='', encoding='utf-8') as csvFile:
#         csvWriter = csv.writer(csvFile)
#         csvWriter.writerow(["Created At", "ID", "Text"])
#         while True:
#             json_response = connect_to_endpoint(url, params)
#             if not json_response['data']:
#                 break
#             for tweet in json_response['data']:
#                 expanded_text = tweet.get('text')
#                 if tweet.get('referenced_tweets'):
#                     for ref_tweet in json_response.get('includes', {}).get('tweets', []):
#                         if ref_tweet['id'] == tweet['referenced_tweets'][0]['id']:
#                             expanded_text = ref_tweet['text']
#                             break
#                 csvWriter.writerow([tweet['created_at'], tweet['id'], expanded_text])
#             if 'next_token' not in json_response['meta']:
#                 break
#             params = get_params(json_response['meta']['next_token'])

def main(username, writer):  
    url = create_url(username)
    params = get_params()
    while True:
        json_response = connect_to_endpoint(url, params)
        if not json_response['data']:
            break
        for tweet in json_response['data']:
            expanded_text = tweet.get('text')
            if tweet.get('referenced_tweets'):
                for ref_tweet in json_response.get('includes', {}).get('tweets', []):
                    if ref_tweet['id'] == tweet['referenced_tweets'][0]['id']:
                        expanded_text = ref_tweet['text']
                        break
            writer.writerow([tweet['created_at'], tweet['id'], expanded_text])
        if 'next_token' not in json_response['meta']:
            break
        params = get_params(json_response['meta']['next_token'])

# if __name__ == "__main__":
#     main()