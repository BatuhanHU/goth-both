import pandas as pd
import ast

df = pd.read_csv('booksummaries.txt', delimiter="\t", names=["WID", "BookName", "Author", "Date", "Tags", "Summary"],
                 encoding="UTF-8")

emptyDf = df[df['Tags'].isna()]
validDf = df.dropna(subset=['Tags'], how='all')

indx = 0
for i in validDf["Tags"]:
    i = ast.literal_eval(i)
    validDf.iat[indx, 4] = i
    indx += 1

validChangedDate = pd.DataFrame(columns=["WID", "BookName", "Author", "Date", "Tags", "Summary"])

counter = 0
for i in range(len(validDf)):
    for j in validDf.iloc[i].Tags:
        validChangedDate.loc[counter] = [validDf.iloc[i].WID, validDf.iloc[i].BookName, validDf.iloc[i].Author, validDf.iloc[i].Date, validDf.iloc[i].Tags[j], validDf.iloc[i].Summary]
        counter += 1

validChangedDate.to_csv('validChangedDate.txt', sep='\t')

