import streamlit as st
import tweepy
from wordcloud import WordCloud
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import nltk
nltk.download('omw-1.4')
nltk.data.path.append('./nltk_data/')
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import text2emotion as te
from gensim.summarization.summarizer import summarize
import docx2txt
from PIL import Image
import pdfplumber
import nltk.corpus
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
from nltk.corpus import stopwords
import streamlit.components.v1 as components

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
import sys
import time
import threading

stopwords2 = stopwords.words("english")

warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)


def getPolarity(text):
    return TextBlob(text).sentiment.polarity


def getemotion(text):
    return te.get_emotion(text)


def getAnalysis(score):
    if score < -0.3:
        return 'Negative'
    elif score > 0.3:
        return 'Positive'
    else:
        return 'Neutral'


def getEmotionResult(happy, angry, surprise, sad, fear):
    lstEmotionLabel = ['happy', 'angry', 'surprise', 'sad', 'fear']
    lstEmotionValue = [happy, angry, surprise, sad, fear]
    if max(lstEmotionValue) == 0:
        return "Neutral"
    maxIndx = lstEmotionValue.index(max(lstEmotionValue))
    return lstEmotionLabel[maxIndx]


def cleanTxt(txt):
    singlelstofwords = [t.lower() for t in txt]
    singlelstofwords = [t.translate(str.maketrans('', '', string.punctuation)) for t in singlelstofwords]
    singlelstofwords = [t.translate(str.maketrans('', '', ',')) for t in singlelstofwords]
    singlelstofwords = [t.translate(str.maketrans('', '', ' ')) for t in singlelstofwords]
    singlelstofwords = [t for t in singlelstofwords if t not in stopwords2]  # stop words
    singlelstofwords = [t for t in singlelstofwords if len(t) > 3]  # remove short words
    userdefinedchars = ["https", "http", "mailto", "via"]
    singlelstofwords = [t for t in singlelstofwords if t not in userdefinedchars]
    wordnet_lemmatizer = WordNetLemmatizer()
    singlelstofcleanedwords = [wordnet_lemmatizer.lemmatize(t) for t in singlelstofwords]
    # cleaned list
    return singlelstofcleanedwords


class AllActions():

    def __init__(self, useraccount):
        consumerKey = 'CAzP9x3jI0FU4cEkICDdE6axb'
        consumerSecret = 'sdYzBHoUme9u7Pw8j454ncoV1DfbhIFxQs3ELbPVsiyV131zKg'
        accessToken = '1314246151197999105-oiKvQdKNW2n7VN9TAYPmhso8xHhV1d'
        accessTokenSecret = 'RZW9guSaFTaUJjHFlDE86rN1Q5aJSea6Fd7wOOl38m5RY'
        # Create the authentication object
        authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)
        # Set the access token and access token secret
        authenticate.set_access_token(accessToken, accessTokenSecret)
        # Creating the API object while passing in auth information
        api = tweepy.API(authenticate, wait_on_rate_limit=True)
        self.posts = api.user_timeline(screen_name=useraccount, count=100, lang="en", tweet_mode="extended")

    def showRecentTweets(self):
        ltweets = []
        for tweet in self.posts[:5]:
            ltweets.append(tweet.full_text)
        return ltweets

    def gen_wordcloud(self):
        df = pd.DataFrame([tweet.full_text for tweet in self.posts], columns=['Tweets'])
        # word cloud visualization
        allWords = ' '.join([twts for twts in df['Tweets']])
        allWords=word_tokenize((allWords))
        allcleanedwords = cleanTxt(allWords)
        allcleanedwords = ' '.join([twts for twts in allcleanedwords])
        wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110,max_words=150).generate(allcleanedwords)
        plt.imshow(wordCloud, interpolation="bilinear")
        plt.axis('off')
        plt.savefig('WC.jpg')
        img = Image.open("WC.jpg")
        return img

    def getBlobPolarityResult(self):
        if (self > 0.5):
            return ("Positive")
        elif (self < -0.5):
            return ("Negative")
        else:
            return ('Neutral')

    def plotAnalysis(self):
        df = pd.DataFrame([tweet.full_text for tweet in self.posts], columns=["Tweets"])
        # # Clean the tweets
        df['Tweets'] = df['Tweets'].apply(lambda x: x.lower())
        df['Tweets'] = df['Tweets'].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))
        df['Polarity'] = df['Tweets'].apply(lambda x: getPolarity(x))
        df['PolarityResult'] = df['Polarity'].apply(lambda x: getAnalysis(x))
        df['Emotion'] = df['Tweets'].apply(lambda x: getemotion(x))
        df['EmotionResult'] = df['Emotion'].apply(
            lambda x: getEmotionResult(x['Happy'], x['Angry'], x['Surprise'], x['Sad'], x['Fear']))
        return df


class Checksemantic():
    def sentimentAnalyse(self, userinput):
        lstLines = sent_tokenize(userinput)
        lstLines = [t.lower() for t in lstLines]
        lstLines = [t.translate(str.maketrans('', '', string.punctuation)) for t in lstLines]
        df = pd.DataFrame(lstLines, columns=['Lines'])
        df['Lines'] = df['Lines'].apply(lambda x: x.lower())
        df['Lines'] = df['Lines'].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))

        df['Polarity'] = df['Lines'].apply(lambda x: getPolarity(x))
        df['PolarityResult'] = df['Polarity'].apply(lambda x: getAnalysis(x))
        df['Emotion'] = df['Lines'].apply(lambda x: getemotion(x))
        df['EmotionResult'] = df['Emotion'].apply(
            lambda x: getEmotionResult(x['Happy'], x['Angry'], x['Surprise'], x['Sad'], x['Fear']))
        df=df[['Lines','PolarityResult','EmotionResult']]
        # df[['Name', 'Qualification']]
        return df

    def genralWordcloud(self, text):
        lstAllWords = word_tokenize(text)
        singlelist = []
        for line in lstAllWords:
            wordlst = word_tokenize(line)
            for word in wordlst:
                singlelist.append(word)
        # clean it
        listofcleanedwords = cleanTxt(singlelist)
        allwords = (" ").join(listofcleanedwords)
        # mask = np.array(Image.open("twordcloud.png"))
        wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allwords)
        plt.imshow(wordCloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig('WC.jpg')
        img = Image.open("WC.jpg")
        return img


class Mainclass():
    def run(self):
        st.title("Explore And Select A Module To Extract Valuable Insights From Text Data...")
        activities = ["---", "Analyse Tweets", "Analyse your text", "Summarization"]
        choice = st.sidebar.selectbox("Modules", activities)

        if choice == "Analyse Tweets":
            st.subheader("Enter a twitter account handle and get the sentiment analysis or wordcloud of tweets")
            raw_text = st.text_input("Enter the exact twitter handle (without @)")
            obj = AllActions(raw_text)
            Analyzer_choice = st.selectbox("Select the result you want to see",
                                           ["5 recent tweets", "WordCloud of last 100 tweets",
                                            "Visualize the sentiment analysis last 100 tweets"])
            if st.button("Next"):
                if Analyzer_choice == "5 recent tweets":
                    st.success("Fetching last 5 Tweets")
                    recent_tweets = obj.showRecentTweets()
                    for t in recent_tweets:
                        st.write(t)
                        st.write("---")
                elif Analyzer_choice == "WordCloud of last 100 tweets":
                    st.success("Generating Word Cloud")
                    img = obj.gen_wordcloud()
                    # st.write(img)
                    st.image(img)
                elif Analyzer_choice == "Visualize the sentiment analysis last 100 tweets":
                    st.success("Generating Visualisation for Sentiment Analysis")
                    df = obj.plotAnalysis()
                    # st.write(df)
                    st.write(sns.countplot(x=df['PolarityResult'], data=df))
                    st.pyplot(use_container_width=True)
                    st.write("---")
                    st.write(sns.countplot(x=df['EmotionResult'], data=df))
                    st.pyplot(use_container_width=True)

        elif choice == "Analyse your text":
            st.subheader("Enter your text data and get the sentiment analysis of it")
            obj2 = Checksemantic()
            uniput = st.text_area("Enter your text:")
            if st.button("Next"):
                st.write(uniput)
                st.success("Sentiment Analysis Result")
                df = obj2.sentimentAnalyse(uniput)
                st.write(df)
                del df;
                st.write("---")
                st.success("WordCloud Result")
                img = obj2.genralWordcloud(uniput)
                st.image(img)

        elif choice == "Summarization":
            st.subheader("Get the summary of a lengthy article")
            activities = ["---", "By typing it", "By uploading a file"]
            choice = st.sidebar.selectbox("Please select how do you want to input the texts:", activities)
            obj2 = Checksemantic()
            if choice == "By typing it":
                usertxt = st.text_area("Enter your text: ")
                wordcount = st.number_input("Enter the number of words to summarize i.e 100")
                if (st.button("Next") and usertxt != ""):
                    st.success("Summary of your text")
                    my_summary = summarize(usertxt, word_count=wordcount)
                    st.write(my_summary)
                    st.write("---")
                    st.write("Word Cloud of your text")
                    img = obj2.genralWordcloud(usertxt)
                    st.image(img)
                else:
                    st.markdown("Please enter your text for summarization!")
            elif choice == "By uploading a file":
                uploaded_file = st.file_uploader("Upload a file", type=['txt', 'docx', 'pdf'])
                wordcount = st.number_input("Enter the number of words to summarize i.e 100")
                if uploaded_file is not None:
                    if st.button("Proceed"):
                        # Check File Type
                        textfile = ""
                        if uploaded_file.type == "text/plain":
                            # raw_text = docx_file.read() # read as bytes
                            # st.write(raw_text)
                            # st.text(raw_text) # fails
                            # st.text(str(uploaded_file.read(), "utf-8"))  # empty
                            textfile = str(uploaded_file.read(),
                                           encoding='unicode_escape')  # works with st.text and st.write,used for futher processing
                            # st.text(raw_text) # Works
                            # st.write(raw_text)  # works
                        elif uploaded_file.type == "application/pdf":
                            # raw_text = read_pdf(docx_file)
                            # st.write(raw_text)
                            try:
                                with pdfplumber.open(uploaded_file) as pdf:
                                    page = pdf.pages[0]
                                    textfile = page.extract_text()
                                    # st.write(page.extract_text())
                            except:
                                st.write("None")
                        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            # Use the right file processor ( Docx,Docx2Text,etc)
                            textfile = docx2txt.process(uploaded_file)  # Parse in the uploadFile Class directory
                            # st.write(raw_text)
                        st.success("Summary of your text")
                        my_summary = summarize(textfile, word_count=wordcount)
                        st.write("---")
                        st.write(my_summary)
                        st.write("---")
                        st.write("Word Cloud of your text")
                        img = obj2.genralWordcloud(textfile)
                        st.image(img)
                        st.write("---")
                        file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type,
                                        "FileSize": uploaded_file.size}
                        st.write(file_details)

        components.html(
            """
            <html>
                <body>            
                    <footer style="color:#0E86D4;" class="page-footer font-small indigo">
                        <div class="footer-copyright text-center py-3">©2021 all rights reserved</br>
                            <b>Designed & Developed By |<a style="color:#FAFBF4" href="https://www.linkedin.com/in/waisyousofi/" target="_blank">Waisullah Yousofi</a>|</b>
                        </div>                  
                    </footer>
                </body>
            </html>
            """,
            height=50, )
if __name__ == "__main__":
    Mainclass().run()
