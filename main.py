import streamlit as st
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
#warnings ignored
import warnings
# imports
import pandas as pd
from matplotlib import pyplot as plt
# %matplotlib inline
import seaborn as sns
from nltk.tokenize import sent_tokenize
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import text2emotion as te
from gensim.summarization.summarizer import summarize
from io import StringIO
# File Processing Pkgs
import docx2txt
from PIL import Image
from PyPDF2 import PdfFileReader
import pdfplumber
import nltk.corpus
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stopwords=nltk.corpus.stopwords.words("english")
import os
import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)


class AllActions(object):
    def initializer(self,useraccount):
#         consumerKey = #confidential
#         consumerSecret = #confidential
#         accessToken = #confidential 
#         accessTokenSecret =#confidential 
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
    # Create a function to clean the tweets
    def cleanTxt(self,txt):
        # split
        text = word_tokenize(txt)
        text=[t.lower() for t in text]
        text=[t.translate(str.maketrans('','',string.punctuation)) for t in text]
        text = [t.translate(str.maketrans('', '',',')) for t in text]
        text = [t.translate(str.maketrans('', '',' ')) for t in text]
        text= [t for t in text if t not in stopwords]#stop words
        userdefinedchars=["https","http","mailto","via"]
        text=[t for t in text if t not in userdefinedchars]
        wordnet_lemmatizer = WordNetLemmatizer()
        text = [wordnet_lemmatizer.lemmatize(t) for t in text]
        return text

    def gen_wordcloud(self):
        df = pd.DataFrame([tweet.full_text for tweet in self.posts], columns=['Tweets'])
        df['Tweets'] = df['Tweets'].apply(lambda x: self.cleanTxt(x))
        allWords = ' '.join(str(twts) for twts in df['Tweets'])
        allWords = allWords.translate(str.maketrans(' ', ' ', "'"))
        mask = np.array(Image.open("twordcloud.png"))
        # wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)
        wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=1000, mask=mask).generate(allWords)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig('WC.jpg')
        img = Image.open("WC.jpg")
        return img

    def genralWordcloud(self,text):
        lstext=self.cleanTxt(text)
        allWords = ' '.join(str(twts) for twts in lstext)
        # mask = np.array(Image.open("twordcloud.png"))
        wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=500).generate(allWords)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig('WC.jpg')
        img = Image.open("WC.jpg")
        return img

    def getSubjectivity(self,text):
        return TextBlob(text).sentiment.subjectivity

    # Create a function to get the polarity
    def getPolarity(self,text):
        return TextBlob(text).sentiment.polarity

    def getAnalysis(self,score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'

    def plotAnalysis(self):
        df = pd.DataFrame([tweet.full_text for tweet in self.posts], columns=['Tweets'])
        # # Clean the tweets
        df['Tweets'] = df['Tweets'].apply(lambda x: x.lower())
        df['Tweets'] = df['Tweets'].apply(lambda x: x.translate(str.maketrans("","",string.punctuation)))
        # # Create two new columns 'Subjectivity' & 'Polarity'
        df['Subjectivity'] = df['Tweets'].apply(lambda x:self.getSubjectivity(x))
        df['Polarity'] = df['Tweets'].apply(lambda x:self.getPolarity(x))
        df['Analysis'] = df['Polarity'].apply(lambda x:self.getAnalysis(x))
        return df


class Checksemantic(object):
    def nltk_sentiment(self,sentence):
        nltk_sentiment = SentimentIntensityAnalyzer()
        sent_score = nltk_sentiment.polarity_scores(sentence)
        return sent_score

    def blob_sentiment(self,sentence):
        sentence = TextBlob(sentence)
        sent_score = sentence.sentiment
        return sent_score

    # classifier emotion
    def emotion_sentiment(self,sentence):
        sent_score = te.get_emotion(sentence)
        return sent_score

    def getNltkResult(self,pos, neu, neg):
        if (pos > neu and pos > neg):
            return ("Positive")
        elif (neg > neu and neg > pos):
            return ("Negative")
        else:
            return ('Neutral')

    # find result
    def getBlobPolarityResult(self,score):
        if (score > 0.5):
            return ("Positive")
        elif (score < -0.5):
            return ("Negative")
        else:
            return ('Neutral')

    # find result
    def getBlobSubjectivityResult(self,score):
        if (score < 0.2):
            return ("Very Objective")
        elif (score < 0.4):
            return ("Objective")
        elif (score < 0.6):
            return ('Neutral')
        elif (score < 0.8):
            return ("Subjective")
        else:
            return ("Very Subjective")

    # find result
    def getEmotionResult(self,happy, angry, surprise, sad, fear):
        lstEmotionLabel = ['happy', 'angry', 'surprise', 'sad', 'fear']
        lstEmotionValue = [happy, angry, surprise, sad, fear]
        if max(lstEmotionValue) == 0:
            return "Neutral"
        maxIndx = lstEmotionValue.index(max(lstEmotionValue))
        return (lstEmotionLabel[maxIndx])

    def process(self,userinput):
        lstLines = sent_tokenize(userinput)
        lstLines = [t.lower() for t in lstLines]
        lstLines = [t.translate(str.maketrans('', '', string.punctuation)) for t in lstLines]
        # using nltk
        nltkResults = [self.nltk_sentiment(t) for t in lstLines]
        # print(nltkResults)
        # using blob
        blobResults = [self.blob_sentiment(t) for t in lstLines]
        # print(blobResults)
        # using blob
        emotionResults = [self.emotion_sentiment(t) for t in lstLines]
        # find result
        # create dataframe
        df = pd.DataFrame(lstLines, columns=['Lines'])
        # dataframe
        # print("\n*** Update Dataframe - Nltk Sentiments ***")
        df['Pos'] = [t['pos'] for t in nltkResults]
        df['Neu'] = [t['neu'] for t in nltkResults]
        df['Neg'] = [t['neg'] for t in nltkResults]
        df['NltkResult'] = [self.getNltkResult(t['pos'], t['neu'], t['neg']) for t in nltkResults]

        # print("\n*** Update Dataframe - TextBlob Sentiments ***")
        df['BlobPolarity'] = [t[0] for t in blobResults]
        df['PolarityResult'] = [self.getBlobPolarityResult(t[0]) for t in blobResults]
        # create dataframe
        df['BlobSubjectivity'] = [t[1] for t in blobResults]
        df['SubjectivityResult'] = [self.getBlobSubjectivityResult(t[1]) for t in blobResults]
        df['Happy'] = [t['Happy'] for t in emotionResults]
        df['Angry'] = [t['Angry'] for t in emotionResults]
        df['Surprise'] = [t['Surprise'] for t in emotionResults]
        df['Sad'] = [t['Sad'] for t in emotionResults]
        df['Fear'] = [t['Fear'] for t in emotionResults]
        df['emotionResult'] = [self.getEmotionResult(t['Happy'], t['Angry'], t['Surprise'], t['Sad'], t['Fear']) for t in
                               emotionResults]
        return df


st.title("This app can help you to visualize the insights of your texts!")
activities = ["...","Explore Tweets", "How my text may sound","Summarization"]
choice = st.sidebar.selectbox("Modules", activities)

if choice == "Explore Tweets":
    st.subheader("Analyze the tweets of your favourite Personalities")
    raw_text = st.text_input("Enter the exact twitter handle (without @)")
    obj = AllActions()
    obj.initializer(raw_text)
    Analyzer_choice = st.selectbox("Select the result you want to see",["5 recent tweets", "WordCloud of last 100 tweets",
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
            st.image(img)
        elif Analyzer_choice == "Visualize the sentiment analysis last 100 tweets":
            st.success("Generating Visualisation for Sentiment Analysis")
            df = obj.plotAnalysis()
            # st.write(df)
            st.write(sns.countplot(x=df["Analysis"], data=df))
            st.pyplot(use_container_width=True)

elif choice == "How my text may sound":
    obj = Checksemantic()
    uniput=st.text_area("Enter your text: ")
    if st.button("Next"):
        st.write(uniput)
        st.success("Sentiment Analysis Result:")
        df = obj.process(uniput)
        st.write(df["NltkResult"][0])
        st.write(df["SubjectivityResult"][0])
        st.write(df["emotionResult"][0])
        del df;

elif choice == "Summarization":
    obj2 = AllActions()
    activities = ["---", "By type", "By uploading a file"]
    choice = st.sidebar.selectbox("How you want to input texts", activities)
    if choice=="By type":
        usertxt=st.text_area("Enter your text: ")
        wordcount = st.number_input("Enter the number of words to summarize i.e 100")
        if (st.button("Next") and usertxt!=""):
            st.success("Summary of your text")
            my_summary = summarize(usertxt, word_count=wordcount)
            st.write(my_summary)
            st.write("---")
            st.write("Word Cloud of your text")
            img = obj2.genralWordcloud(usertxt)
            st.image(img)
        else:
            st.markdown("Please enter your text for summarization!")
    elif choice=="By uploading a file":
        uploaded_file = st.file_uploader("Upload a file", type=['txt','docx','pdf'])
        wordcount = st.number_input("Enter the number of words to summarize i.e 100")
        if uploaded_file is not None:
            if st.button("Proceed"):
                # Check File Type
                textfile=""
                if uploaded_file.type == "text/plain":
                    # raw_text = docx_file.read() # read as bytes
                    # st.write(raw_text)
                    # st.text(raw_text) # fails
                    #st.text(str(uploaded_file.read(), "utf-8"))  # empty
                    textfile = str(uploaded_file.read(),encoding= 'unicode_escape')  # works with st.text and st.write,used for futher processing
                    # st.text(raw_text) # Works
                    # st.write(raw_text)  # works
                elif uploaded_file.type == "application/pdf":
                    # raw_text = read_pdf(docx_file)
                    # st.write(raw_text)
                    try:
                        with pdfplumber.open(uploaded_file) as pdf:
                            page = pdf.pages[0]
                            textfile=page.extract_text()
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

