import tweepy
import configparser
import pandas as pd
from textblob import TextBlob
# read configs
config = configparser.ConfigParser()
config.read('config.ini')
api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']
# print(api_key)

# authentication
# create an authentication handler
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


def search_screen_name(in_screen_name):
    cursor = tweepy.Cursor(api.user_timeline, screen_name=in_screen_name,
                           tweet_mode="extended", count=None).items(100)
    columns = ['Time', 'Tweet', 'Polarity', 'Tweet_Tune']
    data = []
    # create empty variables to
    positive = 0
    negative = 0
    neutral = 0
    for i in cursor:
        polarity = 0
        full_tweet = i.full_text
        analysis = TextBlob(full_tweet)
        polarity += analysis.polarity
        tune = ''
        if polarity > 0:
            # 1 is positive
            tune = 1
            positive += 1
        elif polarity < 0:
            # -1 means negative
            tune = -1
            negative += 1
        else:
            # 0 mean neutral
            tune = 0
            neutral += 1
        # append to data list
        data.append([i.created_at, full_tweet, polarity, tune])

    df = pd.DataFrame(data, columns=columns)

    df.to_csv('user_tweets.csv')
    print(df)
