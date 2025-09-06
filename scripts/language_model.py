import nltk
from collections import defaultdict, Counter

nltk.download("punkt")

corpus = """
When data are noisy, it’s our job as data scientists to listen 
for signals so we can relay it to someone who can decide how to 
act. To amp up how loudly hidden signals speak over the noise 
of big and/or volatile data, we can deploy smoothing algorithms, 
which though traditionally used in time-series analyses, also 
come into their own when applied on other sequential data. 
Smoothing algorithms are either global or local because they take 
data and filter out noise across the entire, global series, or 
over a smaller, local series by summarizing a local or global 
domain of Y, resulting in an estimation of the underlying data 
called a smooth. The specific smoother you use depends on your 
analysis’ goal and data quirks, because as we’ll see below, 
there are trade-offs to consider. Below are a few options, along 
with their intuition, limitations, and formula so you can rapidly 
evaluate when and why to use one over the other.
"""

# 1. TOKENIZATION

# Word-level tokenization (lowercased, keep only words)
words = nltk.word_tokenize(corpus.lower())
words = [w for w in words if w.isalpha()]

# Character-level tokenization
chars = list(corpus.lower().replace(" ", "_"))  # use '_' for spaces

# 2. FUNCTION TO BUILD N-GRAMS
def build_ngrams(tokens, n=2):
    """
    Builds an N-gram probability model from tokens.
    Returns a dictionary mapping history -> next-token probabilities.
    """
    model = defaultdict(Counter)
    for i in range(len(tokens) - n + 1):
        history = tuple(tokens[i:i+n-1])
        next_token = tokens[i+n-1]
        model[history][next_token] += 1

    # Convert counts to probabilities
    for history, counter in model.items():
        total = float(sum(counter.values()))
        for token in counter:
            counter[token] /= total
    return model

# FUNCTION TO GENERATE NEXT TOKEN
def predict_next(model, history, top_k=3):
    """
    Predicts the next token given a history (tuple).
    Returns top_k most probable tokens.
    """
    history = tuple(history)
    if history not in model:
        return None
    return model[history].most_common(top_k)

# Word-level models
unigram_words = build_ngrams(words, n=1)
bigram_words = build_ngrams(words, n=2)
trigram_words = build_ngrams(words, n=3)
quadgram_words = build_ngrams(words, n=4)

# Character-level models
unigram_chars = build_ngrams(chars, n=1)
bigram_chars = build_ngrams(chars, n=2)
trigram_chars = build_ngrams(chars, n=3)
quadgram_chars = build_ngrams(chars, n=4)

# Next Word Prediction
print("Word-level unigram Prediction (history = []):")
print(predict_next(unigram_words, [], top_k=5))

print("\nWord-level Bigram Prediction (history = ['smoothing']):")
print(predict_next(bigram_words, ["smoothing"], top_k=5))

print("\nWord-level Trigram Prediction (history = ['an', 'estimation']):")
print(predict_next(trigram_words, ["an", "estimation"], top_k=5))

print("\nWord-level Quadgram Prediction (history = ['can', 'deploy', 'smoothing']):")
print(predict_next(quadgram_words, ["can", "deploy", "smoothing"], top_k=5))

# Next Character Prediction
print("\nCharacter-level Unigram Prediction (history = []):")
print(predict_next(unigram_chars, [], top_k=5))

print("\nCharacter-level Bigram Prediction (history = ['n']):")
print(predict_next(bigram_chars, ["n"], top_k=5))

print("\nCharacter-level Trigram Prediction (history = ['n','a']):")
print(predict_next(trigram_chars, ["n","a"], top_k=5))

print("\nCharacter-level Quadgram Prediction (history = ['n','a','l']):")
print(predict_next(quadgram_chars, ["n","a","l"], top_k=5))