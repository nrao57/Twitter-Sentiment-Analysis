#Naive Bayes to determine sentiment analysis 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt


data_df = pd.read_csv("trainingdata.txt", sep='\t')
data_df = data_df.sample(frac=1).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(data_df['tweet'], data_df['label'])

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_test_counts = count_vect.transform(X_test)

#Naive Bayes 
gnb = GaussianNB()
y_pred = gnb.fit(X_train_counts.toarray(), y_train).predict(X_test_counts.toarray())


#ROC curve / F Score
fscore = f1_score(y_test, y_pred)
rocscores = roc_curve(y_test, y_pred)
plt.plot(rocscores[0], rocscores[1])
plt.show()
