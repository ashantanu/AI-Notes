# Neural Network Methods for Natural Language Processing
tags: NLP, ML, DL, Book
<!-- TOC -->

- [Neural Network Methods for Natural Language Processing](#neural-network-methods-for-natural-language-processing)
    - [Introduction](#introduction)
    - [Basic Linear models](#basic-linear-models)
    - [Linear to multi layer perceptron](#linear-to-multi-layer-perceptron)
    - [Feed forward NN](#feed-forward-nn)
    - [NN Training](#nn-training)
- [Features for Textual Data](#features-for-textual-data)
    - [Case studies of NLP fetures](#case-studies-of-nlp-fetures)
    - [Things To Read Up](#things-to-read-up)

<!-- /TOC -->

## Introduction
* Challenges in language
    1. *Discrete* : "pizza", "burger" no inherent relationship, it's all a concept in reader's head, can't figure it out directly
    2. *Compositional* : letters -> words -> sentences, sets of rules, and meanings are more than the words/sentences
    3. *Data sparseness* : infinite possibilities, new things keep coming, a dataset can never cover everything

* NN and DL
    * NN -learning parameterised differentiable mathematical functions
    * DL - multiple transformations on data, learns to predict AND correctly represent (encode) data

* DL and NLP
    * words become math objects
    * Architectures
	    * Feed forward NN
        * Convolutional NN
	    * RNN 

## Basic Linear models
* want to map input to output
* cant try all possible functions
* limit to a space/family of functions called *hypothesis class*
    * Problem - "inductive bias* - we are making assumptions about the form of the actual functions
    * benefit - easier to find solution
* For equation $f(\bm{x}) = \bm{x . W + b}$, parameter $\Theta = \bm{W,b }$
* Neaural networks are capable of representing any <b id="borel">Borel-measurable function</b>
* Splitting data
    * Leave One Out : Useful when small dataset (around 100). Train $f_i()$  leaving out a random $x_i$ every time. Train another $f()$ on all data. % of correct $f_i()$ gives an estimate of accuracy of$f()$
    * Held-Out : 80-20 split of data. Use the random 20 for testing
    * Train-Validation-Test : 3 way split. Use validation set to test, tweak, adjust and select model. Leave test set untouched. Prevents again bias of selecting model that was specially adjusted for validation set.


* Models
    * Features extraction : function that maps a real world object (an apartment) to a vector of measurable quantities (price)
    * feature engineering : designing the feature function
    * think of linear models in terms of assigning weights to features - easy to imagine
    * X,Y,W are all representations of the input. Main power of deep learning is the ability to learn good representations
    * One Hot vector : can be called bag-of-a-single-word
    * Bag of words : normalised collection of one-hot over the data. A frequency map of the list of words in your space
    * Continuous bag of words (CBOW) : low dimensional continuous vector
        * $y = xW =$ (sum 
    (one-hot of every word in unput))$\cdot W =$ sum ((every word's one-hot )$\cdot W$)$=$sum(rows of $W$ corresponding to each word)
        * $W$ is a dense word representation/embedding matrix
* Loss functions
     * Our objective is given below. First term is loss, second is regularization

$$\hat{\Theta} = argmin_{\Theta} \left( \frac{1}{n}\sum_{i=1}^{n}L(f(\bm{x}_{i},\Theta),\bm{y_i}) + \lambda R(\Theta) \right)$$
* 
    * Hinge loss(binary) $(\tilde{y},y)=max(0,1-y\cdot \tilde{y})$ where y $\in \{+1,-1\}$ and prediction is the sign of $\tilde{y}$. Thus, it attempts to acieve a correct classification with a margin of atleast 1 (like SVM) (can use another margin $m$ as well) as it also penalises predictions $\tilde{y}$ < 1
    * Hinge (multi-class) $(\hat{y},y)=max(0,1-(\hat{y}_{[t]} - \hat{y_{[k]}}))$ Where t is the correct class of y, and k is the highest scoring category $\ne i$ (y is a vector of scores over all categories). Hingle loss attempts to score the correct class above all other classes with a margin of at least 1 (can use another margin $m$ as well)
    * Log loss $(\hat{y},y)=\log(1+exp(-(\hat{y}_{[t]} - \hat{y_{[k]}})))$
    * Binary cross entropy $(\hat{y},y)=-y\log \hat{y} - (1-y)\log (1-\hat{y})$. Assumes output used sigmoid
    * categorical cross entropy loss $(\hat{y},y)=\sum_{i}-y_i\log \hat{y_{i}}$. Also called negative log likelihood. Assumes softmax is used. Tries to assign more mass to the correct class.
    * [Ranking loss](https://gombru.github.io/2019/04/03/ranking_loss/) : objective is to predict distance between inputs. Ex. matching faces vs mismatching faces. Used with pairs or triples of data points. We want to maximise distance between mismatches, and minimise otherwise.

* Regularization : If we have some incorrect/outliers, it is OK to miss classify a few but fit the majority of data well. Weights can be increased on some features to accomodate outliers, when the shouldn't be.
    * Avoids overfitting
    * makes sure weights don't take extreme value
    * controls the complexity of the function
    * Types : $L_2, L_1$, Elastic net (combo of $L_2, L_1$) and dropout (for NN)

* Stochastic Gradient Descent
    * can use GPUs for minibatch, so faster

## Linear to multi layer perceptron
* XOR - can't be done using linear model
* <b id="kernel-method">Kernel Method</b>
    * Transform non linear inout (XOR) to another space where it can be linearly separated, $\hat{y}=\phi(x)W+b$
    * might have to map to a higher dimension to be linearly separable
    * higher dimensitonal space mapping - more chances of getting a linear separator - but computationally expensive
* **Trainable Mapping**
    * mapping is also learned by using a non linear activation of the input
    * this is what a mulit-layer perceptron is!
    * solve XOR with this now. [solution](https://www.quora.com/How-can-we-design-a-neural-network-that-acts-as-an-XOR-gate)


$$\hat{y}=\phi(x)W+b$$
$$\phi(x)=xW'+b'$$

## Feed forward NN
* Perceptron $\hat{y}=xW+b$
* Theoretically, MLP1 (one hidden layer) can represent all continuous functions on a closed and bounded subset of $\mathbb{R}^n$, but we don't know the size of the hidden layer, or when will our algo reach that solution, how to reach that solution, how easy it is to find given the training data
* Non-linearities : sigmoid (outdated), tanh, hard tanh, Rectifier(ReLU)
* dropout - randomly drop some of the values of hidden layers to prevent the model from relying on some weights too much. Can be achieved by masking the output of hidden layer. Mask is sampled from a bernoulli distribution.
* Similiarity and distance between vectors
    * dot product, cosine
    * trainable
        * sim (bilinear)$(u,v) = uMv$
        * dist $(u,v) = (u-v) M (u-v)$
    * Can also use multi layer perceptron on concatenation of the two vectors
* embedding layer/lookup layer : map x to an embedding. $xE$ when x is one hot vector.

## NN Training
* Backprop Algo: reverse-mode automatic differentiation algo
* Initialization : Algo might give different results each time depending on the initialization and the local minima/saddle point it reaches. Use *xavier initialization* or *He initialization* to control the distribution and magnitude. They are known to work well. Are specific the the type of activation funtion used.
* **Model Ensembles** : use multiple models, then average them out or pick the best, or use some voitng
* Exploding, vanishing gradient
* saturated or dead nuerons : tanh and sigmoid get saturated when input is very large and give small gradients. ReLU gets stuck/dead at 0
* learning rate scheduling : reduce learning rate as the error stops impoving, basically take smaller steps as you get closer to the solution

# Features for Textual Data
* typology of MLP classification problems
    * word : eg. dog. what does it mea, what language, something about the word
    * Texts : Given a text, say about it, the sentiment, spam etc. Called Document Classification
    * Paired Texts : how similar, valid translation, follow logically
    * Word in context : eg. what does *book* in *I want to book a flight* mean. finding named entities, annotations, etc
    * Relation between two words in context : is A the subject of verb B, how is it related etc.
* **Token** : basically a word. When we tokenize a document after considering white spaces and punctuations. New York is a word but two tokens. Don't is one token but two words. Depends on how you tokenize
* indicator (0,1:exists or does not), count - frequency
* features for words : length, capital, etc
    * **lemma** (lemmatizing)- dictionary entry of a word, root. booking,booked,books - book(lemma). Context is important. Well defined rules.
    * **stemming** - pictures, picture - pictur. Depends on aggressiveness. Simpler and easier to implement.
    * Lexical resource - ex - WordNet. external resource that contains info about words
    * distributional information (discussed later)
* features for text
    * bag of words, can also use distribution of length of sentence over number of words, and other such numerical properties
    * Weighing : give weights to bag of words. **TF-IDF** : term frequency $\times\log$ inverse of frequency of the distinct documents this word appears in. This highlights less occuring words.
    * n-grams (discussed later)
* features of word in context
    * Window : immediate context of the word, k surrounding words. as feature, use the word and it's relative position
    * position of the word in the context
    * for relations between words in context, we can look at their relative distance and what appears between them
* linguistic properties as features
    * Use linguisitic rules/properties - syntax, morphology, discourse relations, syntactic trees etc
    * linguistic annotation
        * level 1: assign part of speech. eg. the(DET) boy(Noun) with(Prep) the(Det) black(Adj) shirt(Noun) opened(verb) the(det) door(noun) with(prep) a(det) key(noun)
        * level 2: mark syntactic chunk boundaries. ex. [noun phrase - the boy] [prep phrase - with] [noun phrase - the black shirt]....
        * constituency tree/phrase-structure tree: nested labeled bracketing over the sentence indicating the hierarchy of syntactic units. *the boy with the black shirt* is a noun phrase which itself is made of smaller units
         ![](images/DL_for_NLP-syntactic_tree.png)
        * dependency tree: each word in the sentence is a modifier of another word which is called head. Each word in the sentence is headed by another sentence word, except for the main word, usually a verb, which is the root of the sentence and is headed by a special “root” node. This makes explicity the modification relations and connections between words
         ![](images/DL_for_NLP-dependency_tree.png)
        * semantic role labelling 
        ![](images/DL_for_NLP-semantic_role_labelling.png)
        * Discourse relation - between sentences. eg. Elaboration, contradiction, cause effect etc.
        * anaphora - coreference resolution. eg. *The boy opened the door. It wasn't locked and he smiled.* what do *it* and *he* refer to.
        * these concepts can be implemented as a part of the pipeline, or in a multi-task setup and improve the performance
    * Combination features
        * knowing features that occur in combination is also important. Above things will tell us document has the word *X appears at -1 position* and *Y appears at +3 position* which is not the same as knowing *X appears at -1 and Y appears at +3*. Ex. Paris, Hilton, and Paris Hilton are very different things. Fortunately, NN can take care of this to a good extent, linear models can't (and we have to hand craft them)
        * ngrams - consecutive word sequences of length n. Bag of bigrams has more info, but not all (eg. *of the*), but we let that network (and regularization) will take care of the weights
        * simple NN can't learn ngrams on their own so we need to give them. OR give encode positional info also. CNN, Bi-RNN can figure out these things.
    * **distributional hypothesis** of language : meaning of a word can be inferred from the contexts in which it is used (can use occurence pattern from contexts to know about words)
        * clustering based methods
        * embedding based

## Case studies of NLP fetures
* Make use of all signal in data by
    * giving model direct access to them by use of feature-engineering
    * by designing the network architecture to expose the needed signals
    * by adding them as an additional loss signals when training the models
* Document classification-Language Identification : bag of letter bigrams (core feature)
* embedding detection : byte level bigram
* Document classification-Topic classification : bag of words, with bag of bigrams. Can replace words with their lemmas (if lack of data). Use word embeddings instead of one-hot. TF-IDF. Word indicators instead of frequency.
* Document classification-Authorship Attribution : find author's age, gender, name, or something else. Need to look at stylistic properties rather than the content.

## Things To Read Up
* [Borel Measurable Functions](#borel)
* [Ranking Loss](https://gombru.github.io/2019/04/03/ranking_loss/)
* [Kernel methods and SVM](#kernel-method)
* adaptive learning rate algorithms, RMS Prop, Adam, AdaDelta (Deeplearning lectures)
* SGD plus momentum (Deeplearning lectures)