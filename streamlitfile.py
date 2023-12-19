import numpy as np
import pandas as pd
import streamlit as st
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words('english')

from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import pipeline
classifier = pipeline("zero-shot-classification")



tf_idf_vectorizer = pickle.load(open('updated/vectorizer.pkl','rb'))
knn = pickle.load(open("updated/KNNModel.pkl",'rb'))

st.write(type(tf_idf_vectorizer))

def cleantext(text):
    import re

    #Convert to lower case
    text=text.lower()

    # Get rid of the punctuations
    text_without_punc = re.sub(r'[^\w\s]', '', text)

    # Get rid of the numbers
    text_without_num=re.sub(r'[0-9]', '', text_without_punc)

    #Remove Stopwords
    text_without_sw = [t for t in text_without_num.split() if t not in stop_words]

    # Find the roots
    lemmatized= [WordNetLemmatizer().lemmatize(t) for t in text_without_sw]

    #Join words again
    return " ".join(lemmatized)

def predictEventType(newdata):
    text=newdata['Headline'] + " " + newdata['Abstract'] + " " + newdata['First Part']
    text=cleantext(text)
    st.write(tf_idf_vectorizer)
    vector=tf_idf_vectorizer.transform(text)
    vector=vector.toarray()
    return knn.predict(vector)

def predictSubCategories(newdata):
  event=predictEventType(newdata)[0]
  newdata=newdata['Headline'] + " " + newdata['Abstract'] + " " + newdata['First Part']
  flsub=classifier(newdata,
                   candidate_labels=df[df['Event Category']==event]['1st Level Sub Category'].unique()
    )
  fl=flsub["labels"][np.argmax(flsub["scores"])]

  if df[(df['Event Category']==event) & (df['1st Level Sub Category']==fl)]['2nd Level Sub Category'].nunique()>0:
    slsub=classifier(newdata,
                    candidate_labels=df[(df['Event Category']==event) & (df['1st Level Sub Category']==fl)]['2nd Level Sub Category'].unique()
      )
    sl=slsub["labels"][np.argmax(slsub["scores"])]
  elif df[(df['Event Category']==event) & (df['1st Level Sub Category']==fl)]['2nd Level Sub Category'].nunique()==1:
    sl=df[(df['Event Category']==event) & (df['1st Level Sub Category']==fl)]['2nd Level Sub Category'].unique()
  else:
    sl=np.NaN
    return event, fl, sl, np.NaN



  if df[(df['Event Category']==event) & (df['1st Level Sub Category']==fl) & (df['2nd Level Sub Category']==sl)]['3rd Level Sub Category'].nunique()>1:
    tlsub=classifier(newdata,
                    candidate_labels=df[(df['Event Category']==event) & (df['1st Level Sub Category']==fl) & (df['2nd Level Sub Category']==sl)]['3rd Level Sub Category'].unique()
    )
    tl=tlsub["labels"][np.argmax(tlsub["scores"])]
  elif df[(df['Event Category']==event) & (df['1st Level Sub Category']==fl) & (df['2nd Level Sub Category']==sl)]['3rd Level Sub Category'].nunique()==1:
    tl=df[(df['Event Category']==event) & (df['1st Level Sub Category']==fl) & (df['2nd Level Sub Category']==sl)]['3rd Level Sub Category'].unique()[0]
  else:
    tl=np.NaN
  return event, fl, sl, tl




st.title("Technical Task")

headline=st.text_input("Headline")
abstract=st.text_input("Abstract")
firstPart=st.text_input("First Part")


newdata=pd.DataFrame({"Headline":[headline],"Abstract":[abstract],"First Part":[firstPart]}).iloc[0]
event, fl, sl, tl=predictSubCategories(newdata)
print("event:",event)
print("1st Level Sub Category:",fl)
print("2nd Level Sub Category:",sl)
print("3rd Level Sub Category:",tl)


st.write("test text")