# -*- coding: utf-8 -*-
"""
Sentiment Analysis of Social Media Conversations

This script performs sentiment analysis on social media data (specifically Twitter in this example)
using the VADER lexicon. It fetches tweets based on a search term, analyzes their sentiment,
and provides a basic visualization of the sentiment distribution.

Requirements:
    - Python 3.x
    - Libraries: nltk, tweepy, pandas, matplotlib, seaborn

Installation:
    pip install nltk tweepy pandas matplotlib seaborn

Twitter API Setup:
    1. Create a Twitter Developer account (developer.twitter.com).
    2. Create an app to get your API keys and tokens.
    3. Replace the placeholder credentials below with your actual keys and tokens.

Usage:
    1. Ensure you have the required libraries installed.
    2. Replace the Twitter API credentials in the script.
    3. Define the `search_term` and `num_tweets` as desired.
    4. Run the script. It will print the head of the sentiment DataFrame and
       display a bar chart of the sentiment distribution.

Output:
    - Prints the first few rows of a Pandas DataFrame containing the tweet text and
      its positive, negative, neutral, and compound sentiment scores, as well as
      an overall sentiment label.
    - Displays a bar chart showing the count of positive, negative, and neutral tweets.
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tweepy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Download VADER lexicon if you haven't already (run this once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')

# --- Twitter API Credentials ---
# Replace with your actual Twitter API credentials
consumer_key = "YOUR_CONSUMER_KEY"
consumer_secret = "YOUR_CONSUMER_SECRET"
access_token = "YOUR_ACCESS_TOKEN"
access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"

def authenticate_twitter_api():
    """Authenticates with the Twitter API using provided credentials."""
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api

def fetch_tweets(api, search_term, num_tweets=100):
    """Fetches tweets based on the given search term."""
    try:
        tweets = tweepy.Cursor(api.search_tweets, q=search_term, lang="en", tweet_mode='extended').items(num_tweets)
        tweet_texts = [tweet.full_text for tweet in tweets]
        print(f"Retrieved {len(tweet_texts)} tweets for '{search_term}'.")
        return tweet_texts
    except tweepy.TweepyException as e:
        print(f"Error fetching tweets: {e}")
        return []

def analyze_sentiment(texts):
    """Analyzes the sentiment of a list of texts using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    sentiment_data = []
    for text in texts:
        vs = analyzer.polarity_scores(text)
        sentiment_data.append({
            'text': text,
            'positive': vs['pos'],
            'negative': vs['neg'],
            'neutral': vs['neu'],
            'compound': vs['compound']
        })
    return pd.DataFrame(sentiment_data)

def get_sentiment_label(compound_score):
    """Categorizes sentiment based on the compound score."""
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def visualize_sentiment(sentiment_df, search_term):
    """Visualizes the sentiment distribution."""
    sentiment_counts = sentiment_df['sentiment'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title(f"Sentiment Analysis of Tweets for '{search_term}'")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Tweets")
    plt.show()

if __name__ == "__main__":
    # --- Configuration ---
    search_term = "#AI"  # Define the hashtag or keyword to search for
    num_tweets = 200     # Define the number of tweets to retrieve

    # --- Main Execution ---
    twitter_api = authenticate_twitter_api()
    if twitter_api:
        tweets = fetch_tweets(twitter_api, search_term, num_tweets)
        if tweets:
            sentiment_df = analyze_sentiment(tweets)
            sentiment_df['sentiment'] = sentiment_df['compound'].apply(get_sentiment_label)
            print("\nSentiment Analysis Results:")
            print(sentiment_df.head())
            visualize_sentiment(sentiment_df, search_term)
