from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
import pandas as pd
import ast
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
en_stops = set(stopwords.words('english'))


def stopword(all_words):
    video_sentence = []
    for word in all_words:
        if word not in en_stops and word.isalpha():
            video_sentence.append(word.lower())
    return video_sentence


df = pd.read_csv('validChangedDate.txt', delimiter="\t", names=["WID", "BookName", "Author", "Date", "Tags", "Summary"],
                 encoding="UTF-8")

df = df[["Tags", "Summary"]]
df["CategoryId"] = ""
emptyDf = df[df['Tags'].isna()]
validDf = df.dropna(subset=['Tags'], how='all')

# delete soon


indx = 0
unique_value = []

for i in validDf.values:
    unique_value.append(i[0])
    word_tokens = word_tokenize(i[1])
    filtered_sentence = stopword(word_tokens)
    validDf.iat[indx, 1] = " ".join(filtered_sentence)

    indx += 1
unique_value = set(unique_value)

unique_value_dict = {k: v for v, k in enumerate(unique_value)}

indx = 0
for i in validDf.values:
    validDf.iat[indx, 2] = unique_value_dict.get(i[0])
    indx += 1

X_train, X_test, y_train, y_test = train_test_split(validDf.Summary, validDf.CategoryId, random_state=0)
count_vect = CountVectorizer(min_df=5)
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# print(y_train.to_numpy(dtype=int))
clf = MultinomialNB().fit(X_train_tfidf, y_train.to_numpy(dtype=int))

"""print("wait")
print(clf.predict(count_vect.transform(["This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."])))
print("wait2")"""
y_pred = clf.predict(count_vect.transform(X_test))
"""print(y_pred)
print(y_test.values)
print(len(y_pred), len(y_test.values))"""
# check how many elements are equal in two numpy arrays python
predicted_num = np.sum(y_test.values == y_pred)
print("Accuracy:", predicted_num / len(y_pred))

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

#
"""RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
LinearSVC(),  # Linear Support Vector Classification.
MultinomialNB(),  # Naive Bayes classifier for multinomial models"""
models = [
    LogisticRegression(random_state=0),
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),  # Linear Support Vector Classification.
    MultinomialNB(),  # Naive Bayes classifier for multinomial models"""
]

CV = 10
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, X_train_tfidf.toarray(), y_train.to_numpy(dtype=int), scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
print(cv_df.groupby('model_name').accuracy.mean())

import seaborn as sns

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
