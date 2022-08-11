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

global count


def search_tweets_by_keywords(in_keyword, in_count):
    global count
    count = in_count
    # count = 100
    # create a hashtag_list
    keyword = in_keyword
    # keyword = 'dress'

    # stream by keyword list
    # convert the tweets to csv file
    columns = ['Time', 'Username', 'Tweet', 'Polarity', 'Tweet_Tune']
    data = []
    # -filter:retweets will remove all the retweets from the search
    cursor = tweepy.Cursor(api.search_tweets, q=keyword+' -filter:retweets',
                           tweet_mode="extended", lang='en').items(count)
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
        data.append([i.created_at, i.user.screen_name, full_tweet, polarity, tune])

    df = pd.DataFrame(data, columns=columns)

    df.to_csv('tweet_list.csv')
    print(df)
