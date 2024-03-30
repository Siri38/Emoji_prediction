import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline
import pickle

st.title("Emotions prediction using ML")

text=st.text_input("Enter the text")

model=pickle.load(open(r"C:\Users\Admin\ML_practice\estomator.pkl",'rb'))
st.button("submit")
result=model.predict([text])

if result=='anger':
    st.image('https://i.pinimg.com/474x/ba/c1/03/bac103a791f81dce23cd407086081f77.jpg')
elif result=='love':
    st.image('https://t3.ftcdn.net/jpg/01/92/74/38/360_F_192743872_5lfg1X3UxQYrdQn7lTHliylYS1lF7fl3.jpg')
elif result=='joy':
    st.image('https://cdn.cdnparenting.com/articles/2020/02/21182119/627986495.webp')
elif result=='sad':
    st.image('https://media.tenor.com/X8Q2Vlv-DcsAAAAM/funny-baby.gif')
elif result=='surprise':
    st.image('https://media.tenor.com/XiivI5ETeFgAAAAj/woah-shocked.gif')
elif result=='fear':
    st.image('https://kidsfirstpediatrics.com/wp-content/uploads/2017/09/scared-kids.jpg')
st.write(result)
