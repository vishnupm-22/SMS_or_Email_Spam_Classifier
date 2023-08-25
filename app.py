import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000)
s = PorterStemmer()


def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()


  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()
  for i in text:
    y.append(s.stem(i))

  return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

# 1. preprocess

  transform_SMS = transform_text(input_sms)
# 2. vectorize
  vector_input = tfidf.transform([transform_SMS])  # passing into the List
# 3. predict
  result = model.predict(vector_input)[0]
# 4. Display

  if result == 1:
      st.header("Spam")
  else:
      st.header("Not Spam")




