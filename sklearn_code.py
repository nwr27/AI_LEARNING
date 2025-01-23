from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


data_tweets = [
  "Saya suka belajar AI",
  "AI itu menarik",
  "Belajar bahas pemrograma Python",
  "Ini bukan tentang AI",
]
labels = [1,1,1,0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data_tweets)

model = MultinomialNB()
model.fit(X, labels)

test_data = ["Saya sedang belajar AI"]
test_vector = vectorizer.transform(test_data)
prediction = model.predict(test_vector)
print("Prediksi : ", prediction) 
