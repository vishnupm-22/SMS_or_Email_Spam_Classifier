
## **SMS/EMAIL SPAM DETECTION**
**problem statement**:
* Develop an adaptive spam classification system using advanced machine learning techniques to tackle the significant challenges arising from unwanted emails and text messages.

**Steps** :
* Data Cleaning
* EDA
* Text Preprocessing
* Text Vectorization 
* Model Building and Evaluation
* Model Deployement

Data set contains 5000 + points with features v1 : ham/spam categorical column, v2 : text messages, Others : NaN .

**Did exploratory data analysis to gain some insights on data**:
* **Univariate Analysis**:
  * from bar and pie plot : distribution of data is imbalanced.
  * Hist plot : Explored ham & spam message length distribution & concluded as spam message tends to have more letters & words than ham message.
* **Bivariate Analysis**:
  * Pair Plot : linear relationship between the num-words and num_characters and data contains few outliers
  * heat map : Here the target & num_char colms,num_sentences & num_words are Highly positively correlated
 
**Text Preprocessing**:
* lowercasing the data ensures that the same word in different cases is treated as a single token.
 * made use of NLTK library to tokenize the text to words.
 * Removed special characters,stop words & inflected words to simplify the text.
 * generated a word cloud from the transformed text & also visualized most_common_words of spam and ham in bar plot.
   
Handled imbalanced dataset by applying under-sampling technique .
Used sklearn Count Vectorizer method and TfidfVectorizer class for easily transforming a collection of text documents into a  matrix.

**Model Building and Evaluation**:
* Impleted Naive bayes classifier and evaluated   these models with an accuracy_score precision_score & Confusion_matrix.
* Binomial naive bayes classifier  performs well with an accuracy : 0.95 & precision : 1.0

