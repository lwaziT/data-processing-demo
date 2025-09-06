import nltk # Import python library that is used for NLP and text analytics.
from nltk.tokenize import TreebankWordTokenizer # Import TreebankWordTokenizer class to help us tokenize sentences into words.
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns

from main import tokenize

with open("C:/Users/tobos/OneDrive/Desktop/UNISA 2025/COS4861_NLP/Assignments/Assignment 2/Data preprocessing pipeline/data/Corpus.txt", "r") as file:
    content = file.read()

if __name__ == '__main__':

    sentences, words = tokenize(content)
    # Word frequency distribution
    word_freq = FreqDist(words)
    # Sentence length distribution
    sentence_lengths = [len(nltk.word_tokenize(s)) for s in sentences]

    # Plot top 10 most common words
    common_words = word_freq.most_common(10)
    labels, counts = zip(*common_words)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(labels), y=list(counts), palette="viridis")
    plt.title("Top 10 Most Common Words")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.histplot(sentence_lengths, bins=5, kde=False, color="blue")
    plt.title("Sentence Length Distribution")
    plt.xlabel("Number of words per sentence")
    plt.ylabel("Number of sentences")
    plt.show()