import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Sample sentence
example_sentence = "This is a sample sentence, showing off the stop words filtration."

# Tokenize the sentence
word_tokens = word_tokenize(example_sentence)

# Get the list of stopwords in English
stop_words = set(stopwords.words('english'))

# Filter out the stopwords
filtered_sentence = [word for word in word_tokens if word.lower() not in stop_words]

print("Original Sentence:", example_sentence)
print("Tokenized Words:", word_tokens)
print("Filtered Sentence:", filtered_sentence)
