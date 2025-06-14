from joblib import dump
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from nltk.stem import PorterStemmer
import csv
import os

nltk.download('stopwords')

intents = []
queries = []
responses = []

for file in os.listdir("intents/"):
    if file.endswith(".csv"):
        intent = file[:-4]

        with open("intents/"+file, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None)

            for row in csv_reader:
                intents.append(intent)
                queries.append(row[0])
                responses.append(row[1])

dump(intents, "data/intents.joblib")
dump(queries, "data/queries.joblib")
dump(responses, "data/responses.joblib")

stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_sw_words(doc):
    include = {
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself',
    'yourselves', 'no', 'not', 'yes'
    }

    stop_words = set(stopwords.words('english')) - include
    return(stemmer.stem(w) for w in analyzer(doc) if w.isalpha() and w.lower() not in stop_words)

count_vect = CountVectorizer(analyzer=stemmed_sw_words)
X_train_counts = count_vect.fit_transform(queries)
dump(count_vect, 'models/count_vect.joblib')

tf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
dump(tf_transformer, 'models/tf_transformer.joblib')
dump(X_train_tf, 'models/X_train_tf.joblib')