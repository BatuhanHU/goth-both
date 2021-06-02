import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('booksummaries.txt', delimiter="\t", names=["WID", "BookName", "Author", "Date", "Tags", "Summary"],encoding="UTF-8")

df = df[["Tags", "Summary"]]

emptyDf = df[df['Tags'].isna()]
validDf = df.dropna(subset=['Tags'], how='all')
validDf = validDf[:1000]
indx = 0
unique_value = []
for i in validDf.values:
    tag_value = list(ast.literal_eval(i[0]).values())
    unique_value.extend(tag_value)
    validDf.iat[indx, 0] = " ".join(tag_value)

    summary_value = i[1].lower()
    validDf.iat[indx, 1] = summary_value

    indx += 1
unique_value = set(unique_value)
X_train, X_test, Y_train, Y_test = train_test_split(validDf.Summary, validDf.Tags, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer(min_df=4)
vectorizer.fit(X_train)

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

"""Y_vectorizer = CountVectorizer(min_df=1)
Y_vectorizer.fit(Y_train)

Y_train = Y_vectorizer.transform(Y_train)
Y_test = Y_vectorizer.transform(Y_test)"""

classifier = LogisticRegression(multi_class="multinomial", max_iter=1000)

classifier.fit(X_train, Y_train)
score = classifier.score(X_test, Y_test)

print("Accuracy:", score)