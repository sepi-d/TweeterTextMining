from nltk import FreqDist
from textblob import TextBlob
import streamlit as st
import api_client as tdu
from PIL import Image
import re
import numpy as np
import pandas as pd
import seaborn as sns
import nltk
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import warnings
import twitter_api as tapi
import twitter_id as tid  # Twitter user screen name class


# Logo image
image = Image.open('tweet.jpg')
# page title
st.title("Dana Data Tweet Analyzer")
# display the twitter image
st.image(image)
# st. write("""
# ## you can start to search words in the live tweets
# """)
# create a side bar
st.sidebar.title('Twitter text mining')
st.sidebar.title("Please Select one of the following")
# variable = side bar
# first one empty to have
choice = st.sidebar.selectbox(
    "How do you want to search for data in twitter?",
    ("Home Page", "Search tweets by keyword", "Analyse tweets by user's Screen Name", "Stream Live tweets by keywords"))
# search tweets by keywords using class twitter_api
if choice == "Search tweets by keyword":
    st.subheader("Search Tweets by Keywords")
    with st.form("Twitter_form"):
        st.write("""
        ##### please enter a single word or single hashtags bellow
        if you want to search for hashtag please include # before the word""")
        # take keyword as input from user
        search_keyword = st.text_input("Please Enter a word or hashtag")
        count = st.selectbox("Please chose the number of tweets you want to search?", ('100','400', '1000', '2000'))
        submitted = st.form_submit_button("Submit")
        if submitted:
            # search tweets by given word and number of tweets(api_client class)
            tapi.search_tweets_by_keywords(search_keyword, int(count))
            st.write("The search is now complete")

    warnings.filterwarnings('ignore')
    # read data frame
    df = pd.read_csv('tweet_list.csv')

    # function remove pattern in the input text
    def remove_pattern(input_text, pattern):
        r = re.findall(pattern, input_text)
        for word in r:
            input_text = re.sub(word, "", input_text)
            
        return input_text

    def lower_case(input_text):
        r = re.findall("([A-Z]+)", input_text)
        for i in r:
            input_text = re.replace(i, i.lower())
            return input_text

    # remove twitter handles
    # create a new colum
    df['clean_tweet'] = np.vectorize(remove_pattern)(df['Tweet'], "@[\w]*")

    df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")
   # df['clean_tweet'] = lower_case(df['clean_tweet'])
    # remove url(hyperlinks)
    df['clean_tweet'] = np.vectorize(remove_pattern)(df['clean_tweet'], "http\S+")
    # remove RT(Retweets)
    df['clean_tweet'] = np.vectorize(remove_pattern)(df['clean_tweet'], "RT")
    # create a new dataframe in csv
    df.to_csv("clean.csv")

    # tokenize and remove stop words also stemming
    tokenized_tweets = []
    tweet_sentences = []
    for tweet in df['clean_tweet']:
        # tokenize tweets
        tokenized_tweet = word_tokenize(tweet)
        stop_words = set(stopwords.words("english"))
        ps = PorterStemmer()
        filtered_sentence = []
        for w in tokenized_tweet:
            stem_word = ps.stem(w)
            if stem_word not in stop_words:
                filtered_sentence.append(w)
        tokenized_tweets.append(filtered_sentence)
        tweet_sentences.append(filtered_sentence)

    # combine all the words from the same index into sentences
    for i in range(len(tweet_sentences)):
        tweet_sentences[i] = " ". join(tweet_sentences[i])
    # print(tokenized_tweets)
    # for i in tokenized_tweets:
    #     print(i)
    st.write("To visualize the most frequent words used in the tweets "
             " please press the button bellow")
    if st.button("Visualize Frequent words"):
        words = " ".join([sentence for sentence in tweet_sentences])
    # print(all_words)
        wordcloud = WordCloud(width=1000, height=800, random_state=50, max_font_size=100).generate(words)
        plt.figure(figsize=(80, 60))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.axis('off')
        plt.savefig('WC1.jpg')
        img_keyword = Image.open("WC1.jpg")
        st.image(img_keyword)

    def getTextSubjectivity(tweets):
        return TextBlob(tweets).sentiment.subjectivity


    def getTextPolarity(doc):
        return TextBlob(doc).sentiment.polarity


    df['Subjectivity'] = df['clean_tweet'].apply(getTextSubjectivity)

    df['Polarity1'] = df['clean_tweet'].apply(getTextPolarity)

    def getTextAnalysis(pol):
        if pol > 0:
            return "Positive"
        elif pol == 0:
            return "Neutral"
        else:
            return "Negative"
    # create data frame for score
    df['Score'] = df['Polarity1'].apply(getTextAnalysis)

    st.write("To view analysis of the Polarity of the extracted Tweets Please press the button bellow")

    if st.button("View Bar-Chart"):
        dfs = df['Score'].value_counts()
        freqword = FreqDist()
        for words in dfs:
            freqword[words] += 1
        dfs = dfs[:20]
        plt.figure(figsize=(10,5))
        sns.barplot(dfs.values, dfs.index, alpha=0.8)
        plt.title("Tweets Polarity ")
        plt.ylabel('Tweet Analysis', fontsize=12)
        plt.xlabel("Number of Tweets", fontsize=12)
        # save the image of barchart and run it
        plt.savefig('keyword_bar2.jpg')
        img_bar_key_pol = Image.open("keyword_bar2.jpg")
        st.image(img_bar_key_pol)
    # function to extract hashtags

    def hashtag_extract(tweets):
        hashtag_list = []
        for word in tweets:
            # use re to find hashtags in tweets
            hts = re.findall(r"#(\w+)", word)
            hashtag_list.append(hts)
        return hashtag_list

    pos_hashtag = sum(hashtag_extract(df['clean_tweet'][df['Tweet_Tune'] == 1]), [])
    # # pos_hashtag = (pos_hashtag,[])
    neg_hashtag = sum(hashtag_extract(df['clean_tweet'][df['Tweet_Tune'] == -1]), [])
    neut_hashtag = sum(hashtag_extract(df['clean_tweet'][df['Tweet_Tune'] == 0]), [])
    #
    # create a barchar of the 20 most used words
    dfs = pd.DataFrame(tokenized_tweets)
    dfs = dfs[0].value_counts()
    freqword= FreqDist()
    for words in dfs:
        freqword[words] += 1
    dfs = dfs[:20]
    plt.figure(figsize=(10,5))
    sns.barplot(dfs.values, dfs.index, alpha=0.8)
    plt.title("Top Words Overall")
    plt.ylabel('Word from tweet', fontsize=12)
    plt.xlabel("Count of Words", fontsize=12)
    # save the image of barchart and run it
    plt.savefig('bar.jpg')
    img = Image.open("bar.jpg")
    st.write('To View a bar-chart which illustrate the top 20 frequent words used in tweets'
             'please press the button bellow')
    if st.button("most frequent words"):
        st.image(img)

    st.write('To view the hashtags used positive tweets please press the button bellow ')
    if st.button("# in positive hashtags"):
        freq = nltk.FreqDist(pos_hashtag)
        d = pd.DataFrame(
            freq.values(), freq.keys())
        d = d.nlargest(columns=0, n=4)
        st.bar_chart(data=d)

    st.write('To view the hashtags used in the negative tweets please press the button bellow ')
    if st.button("# in negative tweets"):
        freq = nltk.FreqDist(neg_hashtag)
        d = pd.DataFrame(
            freq.values(), freq.keys())
        d = d.nlargest(columns=0, n=4)
        st.bar_chart(data=d)

if choice == "Analyse tweets by user's Screen Name":
    st.subheader("Search Tweets by Keywords")
    with st.form("Twitter_form"):
        st.write("""
        ##### please chose a user screen name  bellow
            """)
        screen_name = st.selectbox("Please chose a user screen name you want to analyse?", ('elonmusk','ladbible', 'ManMetUni', 'BarackObama'))
        submitted = st.form_submit_button("Submit")
        if submitted:
            # search tweets by given word and number of tweets(api_client class)
            tid.search_screen_name(screen_name)
            st.write("Search is now Complete")
    warnings.filterwarnings('ignore')

    df = pd.read_csv('user_tweets.csv')

    # function remove pattern in the input text
    def remove_pattern(input_text, pattern):
        r = re.findall(pattern, input_text)
        for word in r:
            input_text = re.sub(word, "", input_text)
        return input_text


    # remove twitter handles
    # create a new colum
    df['clean_tweet'] = np.vectorize(remove_pattern)(df['Tweet'], "@[\w]*")

    df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")
    # remove url(hyperlinks)
    df['clean_tweet'] = np.vectorize(remove_pattern)(df['clean_tweet'], "http\S+")
    # remove RT(Retweets)
    df['clean_tweet'] = np.vectorize(remove_pattern)(df['clean_tweet'], "RT")
    # create a new dataframe in csv
    df.to_csv("clean_user.csv")

    # tokenize and remove stop words also stemming
    tokenized_tweets = []
    tweet_sentences = []
    for tweet in df['clean_tweet']:
        # tokenize tweets
        tokenized_tweet = word_tokenize(tweet)
        stop_words = set(stopwords.words("english"))
        ps = PorterStemmer()
        filtered_sentence = []
        for w in tokenized_tweet:
            stem_word = ps.stem(w)
            if stem_word not in stop_words:
                filtered_sentence.append(w)
        tokenized_tweets.append(filtered_sentence)
        tweet_sentences.append(filtered_sentence)

    # combine all the words from the same index into sentences
    for i in range(len(tweet_sentences)):
        tweet_sentences[i] = " ". join(tweet_sentences[i])

    # most frequent words used in the tweets

    st.write("To visualize the most frequent words used in the recent tweets by " + str(screen_name) +
             " please press the button bellow")

    if st.button("Visualize tweets"):
        # puts all the words inside one variable
        words = " ".join([sentence for sentence in tweet_sentences])
    # print(all_words)
        wordcloud = WordCloud(width=1000, height=800, random_state=42, max_font_size=100).generate(words)
        plt.figure(figsize=(60, 40))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.axis('off')
        # save the image
        plt.savefig('WC_user.jpg')
        img_user = Image.open("WC_user.jpg")
        # display the image on dashboard
        st.image(img_user)


    def getTextSubjectivity(tweets):
        return TextBlob(tweets).sentiment.subjectivity


    def getTextPolarity(doc):
        return TextBlob(doc).sentiment.polarity


    df['Subjectivity'] = df['clean_tweet'].apply(getTextSubjectivity)

    df['Polarity1'] = df['clean_tweet'].apply(getTextPolarity)

    def getTextAnalysis(pol):
        if pol > 0:
            return "Positive"
        elif pol == 0:
            return "Neutral"
        else:
            return "Negative"

    df['Score'] = df['Polarity1'].apply(getTextAnalysis)
    st.write("To view analysis of the Polarity(Positive/Negative/Neutral) of " + str(screen_name) + "Tweets Please press the button bellow")

    if st.button("View Bar-Chart"):
        dfs = df['Score'].value_counts()
        freqword = FreqDist()
        for words in dfs:
            freqword[words] += 1
        dfs = dfs[:20]
        plt.figure(figsize=(10,5))
        sns.barplot(dfs.values, dfs.index, alpha=0.8)
        plt.title("Tweets Polarity ")
        plt.ylabel('Tweet Analysis', fontsize=12)
        plt.xlabel("Number of Tweets", fontsize=12)
        # save the image of barchart and run it
        plt.savefig('bar1_user.jpg')
        img_bar1_user = Image.open("bar1_user.jpg")
        st.image(img_bar1_user)

    # function to extract hashtags
    def hashtag_extract(tweets):
        hashtag_list = []
        for word in tweets:
            # use re to find hashtags in tweets
            hts = re.findall(r"#(\w+)", word)
            hashtag_list.append(hts)
        return hashtag_list

    pos_hashtag = sum(hashtag_extract(df['clean_tweet'][df['Score'] == 1]), [])
    # # pos_hashtag = (pos_hashtag,[])
    neg_hashtag = sum(hashtag_extract(df['clean_tweet'][df['Score'] == -1]), [])
    neut_hashtag = sum(hashtag_extract(df['clean_tweet'][df['Score'] == 0]), [])

    # create a barchart of the 20 most used words
    dfs = pd.DataFrame(tokenized_tweets)
    dfs = dfs[0].value_counts()
    freqword = FreqDist()
    for words in dfs:
        freqword[words] += 1
    dfs = dfs[:20]
    plt.figure(figsize=(10,5))
    sns.barplot(dfs.values, dfs.index, alpha=0.8)
    plt.title("Top Words ")
    plt.ylabel('Word from tweet', fontsize=12)
    plt.xlabel("Number of Words", fontsize=12)
    # save the image of barchart and run it
    plt.savefig('bar2_user.jpg')
    img_bar2_user = Image.open("bar2_user.jpg")
    st.write("To View a bar-chart which illustrate the top 20 words used in the user's tweets"
             "please press the button bellow")
    if st.button("Most frequent words Bar-Chart"):
        # display image in streamlit
        st.image(img_bar2_user)

# analyze live tweets
if choice == "Stream Live tweets by keywords":
    st.subheader("Analyse live Tweets ")
    with st.form("Twitter_form"):
        st.write("""
        ##### please enter words or hashtags include # before words""")
        search_keyword = st.text_input("Please Enter a word")
        count = st.selectbox("Please chose the number of tweets you want to search?", ('10', '100','400', '1000'))
        submitted = st.form_submit_button("Submit")
        if submitted:
            # search tweets by given word and number of tweets(api_client class)
            tdu.search_tweet(search_keyword, int(count))
            st.write("search is now complete")

    warnings.filterwarnings('ignore')

    df = pd.read_csv('tweet_list1.csv')

    # function remove pattern in the input text
    # function to remove patterns
    def remove_pattern(input_text, pattern):
        r = re.findall(pattern, input_text)
        for word in r:
            input_text = re.sub(word, "", input_text)
        return input_text


    # remove twitter handles(@)
    # create a new colum
    df['clean_tweet'] = np.vectorize(remove_pattern)(df['Tweet'], "@[\w]*")

    df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")
    # remove url(hyperlinks)
    df['clean_tweet'] = np.vectorize(remove_pattern)(df['clean_tweet'], "http\S+")
    # remove RT(Retweets)
    df['clean_tweet'] = np.vectorize(remove_pattern)(df['clean_tweet'], "RT")
    # create a new dataframe in csv
    df.to_csv("clean_stream.csv")

    # tokenize and remove stop words also stemming
    tokenized_tweets = []
    tweet_sentences = []
    for tweet in df['clean_tweet']:
        # tokenize tweets
        tokenized_tweet = word_tokenize(tweet)
        stop_words = set(stopwords.words("english"))
        ps = PorterStemmer()
        filtered_sentence = []
        for w in tokenized_tweet:
            stem_word = ps.stem(w)
            if stem_word not in stop_words:
                filtered_sentence.append(w)
        tokenized_tweets.append(filtered_sentence)
        tweet_sentences.append(filtered_sentence)

    # combine all the words from the same index into sentences
    for i in range(len(tweet_sentences)):
        tweet_sentences[i] = " ". join(tweet_sentences[i])
    # print(tokenized_tweets)
    # for i in tokenized_tweets:
    #     print(i)
    st.write("To visualize the most frequent words used in the tweets "
             " please press the button bellow")
    if st.button("Visualize tweets"):
        words = " ".join([sentence for sentence in tweet_sentences])
    # print(all_words)
        wordcloud = WordCloud(width=1000, height=800, random_state=42, max_font_size=100).generate(words)
        plt.figure(figsize=(60, 40))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.axis('off')
        plt.savefig('WC_tweet_stream.jpg')
        img_stream = Image.open("WC_tweet_stream.jpg")
        st.image(img_stream)
    #   function to extract hashtags


    def hashtag_extract(tweets):
        hashtag_list = []
        for word in tweets:
            # use re to find hashtags in tweets
            hts = re.findall(r"#(\w+)", word)
            hashtag_list.append(hts)
        return hashtag_list


    pos_hashtag = sum(hashtag_extract(df['clean_tweet'][df['Tweet_Tune'] == 1]), [])
    # # pos_hashtag = (pos_hashtag,[])
    neg_hashtag = sum(hashtag_extract(df['clean_tweet'][df['Tweet_Tune'] == -1]), [])
    neut_hashtag = sum(hashtag_extract(df['clean_tweet'][df['Tweet_Tune'] == 0]), [])
    #
    dfs = pd.DataFrame(tokenized_tweets)
    dfs = dfs[0].value_counts()
    freqword= FreqDist()
    for words in dfs:
        freqword[words] += 1
    dfs = dfs[:20]
    plt.figure(figsize=(10,5))
    sns.barplot(dfs.values, dfs.index, alpha=0.8)
    plt.title("Top Words Overall")
    plt.ylabel('Word from tweet', fontsize=12)
    plt.xlabel("Count of Words", fontsize=12)
    # save the image of barchart and run it
    plt.savefig('bar_stream1.jpg')
    img_bar1 = Image.open("bar_stream1.jpg")
    st.write('To View a bar-chart which illustrate the top 20 frequent words used in tweets'
             'please press the button bellow')
    if st.button("most frequent words"):
        st.image(img_bar1)


    def getTextSubjectivity(tweets):
        return TextBlob(tweets).sentiment.subjectivity

    # check the tweets polarity


    def getTextPolarity(doc):
        return TextBlob(doc).sentiment.polarity


    df['Subjectivity'] = df['clean_tweet'].apply(getTextSubjectivity)

    df['Polarity1'] = df['clean_tweet'].apply(getTextPolarity)


    def getTextAnalysis(pol):
        if pol > 0:
            return "Positive"
        elif pol == 0:
            return "Neutral"
        else:
            return "Negative"


    df['Score'] = df['Polarity1'].apply(getTextAnalysis)

    st.write("To view analysis of the Polarity(Positive/Negative/Neutral) of the extracted tweets Please press the button bellow")

    if st.button("Bar-Chart"):
        dfs = df['Score'].value_counts()
        freqword = FreqDist()
        for words in dfs:
            freqword[words] += 1
        dfs = dfs[:20]
        plt.figure(figsize=(10,5))
        sns.barplot(dfs.values, dfs.index, alpha=0.8)
        plt.title("Tweets Polarity ")
        plt.ylabel('Tweet Analysis', fontsize=12)
        plt.xlabel("Number of Tweets", fontsize=12)
        # save the image of barchart and run it
        plt.savefig('bar2_stream.jpg')
        img_bar2_stream = Image.open("bar2_stream.jpg")
        st.image(img_bar2_stream)

        st.write('To view the positive hashtags used in the tweets please press the button bellow ')
        if st.button("positive hashtags analysis"):
            freq = nltk.FreqDist(pos_hashtag)
            d = pd.DataFrame(
                freq.values(), freq.keys())
            d = d.nlargest(columns=0, n=4)
            st.bar_chart(data=d)



