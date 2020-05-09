# Mannings Stanford NLP Lectures
tags: NLP, ML, DL, Lectures, Stanford, Manning


Best to refer to notes [here](http://web.stanford.edu/class/cs224n/index.html#schedule).
# Lecture 1
representing meaning
- denotational semantics - think of meaning as what thing represent, eg. chair. Computationally tough. ex. WordNet - lists out words, meanings, relationships, synonyms etc. Incomplete
- discrete symbols - ex. one hot encoding, but localist representation of the dictionary you have, no way of understanding relationships
- distributional semantics - use context in which it appears. "word's meaning is explained by the words that frequently appear close by". So represent word by a small(compared to dict size) dense word vector

nlp tasks
- information retrieval - Unstructured text to database entries. Ex. creating knowledge base
- question answering. ex. What is the capital of Germany?
- natural language interaction - Understand requests and act on them. Ex. make a reservation
- summarization - condensing a document
- machine translation

**Softmax**
- $softmax(\bm{x})=\frac{exp(\bm{x})}{\sum_i exp(x_i)}$
- hard-max: take the maximum
- max because exp increases weight of the largest value
- soft because doesn't zero out others and gives some probability weight to other values

