from itertools import product
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from collections import Counter  # For vectorization
from nltk.stem.wordnet import WordNetLemmatizer  # For lemmatization
from nltk.tokenize import sent_tokenize, word_tokenize  # For stemming
from nltk.stem import PorterStemmer  # For stemming
from nltk.corpus import stopwords
import string
import nltk
import pandas as pd
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import wordnet

nltk.download('punkt')  # For tokenization
nltk.download('stopwords')  # For stopwords
nltk.download('wordnet')  # For lemmatization
nltk.download('omw-1.4')  # For lemmatization
nltk.download('vader_lexicon')  # For vader

tweeter_database = pd.read_csv('apple-twitter.csv', encoding='utf-8')
tweets = tweeter_database['text'].tolist()

# DEBUG print(tweets)
# DEBUG iterator = 1
# DEBUG for tweet in tweets:
# DEBUG print(f'{iterator}  {tweet}')
# DEBUG iterator += 1


processed_tweets = []

# Processing the tweets
for tweet in tweets:
    # Tokenize tweet
    tokenized_tweet = nltk.word_tokenize(tweet)

    # Filter stop words
    stop_words = set(stopwords.words("english"))

    # Filtering tweet part1
    filtered_tweet_demo = []
    for w in tokenized_tweet:
        if w not in stop_words:
            filtered_tweet_demo.append(w)

    # Adding to our bag of words
    new_bag_words = [',', 'By', 'What', '\'s', '.',
                     '(', ')', 'The', '-', '`', '[', ']', '\'\'', '``']
    stop_words.update(new_bag_words)

    # Filterint tweet part2
    filtered_tweet = []
    for w in filtered_tweet_demo:
        if w not in stop_words:
            filtered_tweet.append(w)

    # Lemmatization
    lemm = WordNetLemmatizer()

    lemmed_words = []
    for w in filtered_tweet:
        lemmed_words.append(lemm.lemmatize(w))

    # Create a sorted words map and add to processed tweets list
    words_map = Counter(lemmed_words)
    sorted_words = sorted(words_map, key=words_map.get, reverse=False)
    processed_tweets.append(sorted_words)


# A A A A A A A A A A A A A A A


# Create and display word clouds for a few most popular tweets and save them
saved_tweets = []
iterator = 1
for tweet in processed_tweets:
    saved_tweets.append(tweet)
    if iterator > 6:
        break
    print(f'{iterator} {tweet}')

    # Create word cloud img
    wordcloud = WordCloud().generate(' '.join(tweet))

    # Display generated img
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    iterator += 1


# B B B B B B B B B B B B B B B B B B


# Check if tweet is positive or negative
def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    if sentiment_dict['compound'] >= 0.05:
        # print("Positive")
        return 0
    elif sentiment_dict['compound'] <= - 0.05:
        # print("Negative")
        return 1
    else:
        return -1


# Group saved tweets by positive and negative
positive_tweets = []
negative_tweets = []
for tweet in saved_tweets:
    if (sentiment_scores(' '.join(tweet))) == 0:
        positive_tweets.append(tweet)
    elif (sentiment_scores(' '.join(tweet))) == 1:
        negative_tweets.append(tweet)
    # elif (sentiment_scores(' '.join(tweet))) == -1:
        # print("Neutral tweet!")


def showWordCloud(someWords):
    tweetString = ' '
    for tweet in someWords:
        tweetString = tweetString + ' '.join(tweet)

     # Create word cloud img
    # print(f'tweetString: {tweetString}')
    wordcloud = WordCloud().generate(tweetString)

    # Display generated img
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# Show positive tweets wordcloud
print("Positive tweets word cloud...")
if len(positive_tweets) > 0:
    print(positive_tweets)
    showWordCloud(positive_tweets)
else:
    print("No positive tweets!")

# Show negative tweets wordcloud
print("Negative tweets word cloud...")
if len(negative_tweets) > 0:
    print(negative_tweets)
    showWordCloud(negative_tweets)
else:
    print("No negative tweets!")


# C C C C C C C C C
# Function to convert
def listToString(s):

    # initialize an empty string
    str1 = " "

    # return string
    return (str1.join(str(s)))


positive_str = listToString(positive_tweets)
negative_str = listToString(negative_tweets)
syn1 = wordnet.synsets(positive_str)
syn2 = wordnet.synsets(negative_str)

sims = []
for sense1, sense2 in product(syn1, syn2):
    d = wordnet.wup_similarity(sense1, sense2)
    sims.append((d, syn1, syn2))

for sim in sims:
    print(sim)
