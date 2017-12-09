from sklearn.feature_extraction.text import CountVectorizer
import gzip

#Create vectorizer object
vectorizer = CountVectorizer(ngram_range=(2,2), stop_words="english")

idToQuestions = {}

titleAndBodies = []
with gzip.open('text_tokenized.txt.gz', 'rb') as f:

    for file_content in f:

        qId, qTitle, qBody = file_content.split("\t")
        idToQuestions = {qId: (qTitle, qBody)}
        titleAndBodies.append(qTitle)
        titleAndBodies.append(qBody)

x = vectorizer.fit_transform(titleAndBodies)

def parseSentence(sentence):
    vectorizer.transform(sentence).toarray()
