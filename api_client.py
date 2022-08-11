import tweepy
import configparser
import pandas as pd
from textblob import TextBlob


config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']
# authentication
# create an authentication handler
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
# create API object pass auth details
api = tweepy.API(auth)

global count


# use stream class
class Linstener(tweepy.Stream):
    # create a list
    tweets_list = []
    # how many tweets we want

    def on_status(self, status):
        self.tweets_list.append(status)
        if len(self.tweets_list) == count:
            self.disconnect()

    def on_error(self, status_code):
        if status_code == 420:
            # returning False in on_data disconnects the stream
            return False


# stream tweet object
def search_tweet(in_keyword, in_count):
    global count
    count = in_count
    stream_tweet = Linstener(api_key, api_key_secret, access_token, access_token_secret)
    # create a hashtag_list
    key_words = [in_keyword]
    # stream by keyword list
    stream_tweet.filter(track=key_words, languages=['en'])
    # create DataFrame

    columns = ['Username', 'Tweet', 'Polarity', 'Tweet_Tune']
    data = []
    # create empty variables to
    positive = 0
    negative = 0
    neutral = 0
    for tweet in stream_tweet.tweets_list:
        full_tweet = ''
        polarity = 0
        if hasattr(tweet, "retweeted_status"):  # Check if Retweet
            try:
                full_tweet = tweet.retweeted_status.extended_tweet["full_text"]
            except AttributeError:
                full_tweet = tweet.retweeted_status.text
        else:
            try:
                full_tweet = tweet.extended_tweet["full_text"]
            except AttributeError:
                full_tweet = tweet.text

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
        data.append([tweet.user.screen_name, full_tweet, polarity, tune])

    print('positive:', positive)
    print('negative:', negative)
    print('neutral:', neutral)

    # create data frame from list
    df = pd.DataFrame(data, columns=columns)
    # convert df to csv file
    df.to_csv("tweet_list1.csv")
    print(df)



