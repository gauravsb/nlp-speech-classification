# Speech Classification

# Design Decisions
## Preprocessing
* **Using start and end tokens <s\> and </s\>**: We decided to mark the start and end of each sentence with special tokens to improve the semantics of our models. This makes it easier to know when a sentence has logically ended, without relying on the period alone. This is helpful because some periods do not mark the end of sentences, as in the word "Jr.". It also provides a token to use as a seed for generating random sequences.

* **Handling of contractions and possessives**: To handle the inconsistent parsing of contractions and possessives in the corpus (e.g. "you 've" vs. "you ' ve" vs.
"you' ve"), we will remove all spaces on either side of all apostrophes, so that contractions will be tokenized as single words (e.g. 
"you've"). This will improve the consistency of our n-gram models because different parsings of the same contraction should count as the
same word. We acknowledge that there are some instances of separated contractions where the apostrophe is still surrounded by letters 
(e.g. "ca n't"), but unfortunately there is not much we can do about this because there is no way to know programmatically that any two 
consecutive words are a contraction that should be combined. These instances will remain as multiple tokens (e.g. "ca" and "n't").

* **Handling of whitespace surrounding periods**: There are two kinds of periods in the corpora: those that mark the end of their sentence and those that do not. By and large, we feel it is a safe assumption that all sentence-ending periods will be surrounded by either a whitespace character or a newline character. Others that are bordered on one or both sides by a letter, another period, etc. are not intended to end sentences. During preprocessing, we will replace the newlines after some periods with spaces, as there is seemingly nothing that is specially indicated by the location of a newline.


## Data Structures
* **Dictionary mapping word types to IDs**: Store mappings of word types to IDs in constant-access data structure (dictionary). Each word type is assigned a unique ID corresponding to its row and column index in the bigram count matrix. This is a clean, computationally inexpensive method of maintaining these mappings.

* **Bigram count matrix**: Store bigram counts in a matrix `M` such that `M[i,j]` is the number of times that the bigram `(t_i t_j)` appears in the corpus, where `t_i` is the word type at index `i` and similarly for `j`. We are using a matrix to store bigram counts because it gives us constant time access and update, and NumPy has many optimized functions for matrix operations, such as summing a row, which can also be used to get unigram counts. Although in the unsmoothed bigram model the matrix will be extremely sparse, having it in a matrix will make smoothing easier.

* **Dictionary mapping IDs to word types**: Store string expressions of word types in a dictionary, indexed by their word type ID. This will allow constant time translation from token ID back to the token, which will be used when generating sentences.

## Training n-grams

# Random Sentences
### Unigram Examples
#### Obama
1. <s\> modernize some densely kennedy armed will to arise in plane pay to . of , on scratched even , </s\>
2. <s\> nuclear </s\>
3. <s\> one the individuals sick , tonight own international community deficits with not breathe . than are do were of to strongly leak with signed them price <s\> joe american you a if costs from safe and any on reach makes bring and . and . end politics a broker 1999. millions and </s\>
4. <s\> the was our i reforms pride this may underscores the secure of let’s on your natural right other on a snow interest peacefully home we ceiling little , know to enforced health mother . law a of them around has send and some of , . iran through </s\>
5. <s\> are </s\>

#### Trump
1. <s\> do </s\>
2. <s\> magazine </s\>
3. <s\> , <s\> to i <s\> – </s\>
4. <s\> he’s be – people promotion think <s\> should in <s\> women <s\> and like we believe to </s\>
5. <s\> </s\>

### Bigram Examples
#### Obama
1. <s\> and kept our destiny . </s\>
2. <s\> so has a way into the muslim and our power grids , and grenades but we should . </s\>
3. <s\> not prevent the wellbeing of us at a recount -- science that he was never identified over 500 regulatory framework that producing ground-breaking technologies should have not subject to be able to call our relationship . </s\>
4. <s\> oh , alternatively , america is life-changing . </s\>
5. <s\> now , israel at noaa , the situation . </s\>

#### Trump
1. <s\> so they would have a speech was taken care of which is hard . </s\>
2. <s\> i love that if the state building . </s\>
3. <s\> so , and disarray , real jobs have a lot of trouble . </s\>
4. <s\> you’re going to . </s\>
5. <s\> you get bergdahl . </s\>

### Analysis
The bigram model is clearly the better model for creating a probability distribution that yields more coherent sentences. The unigram models for both corpora result in more 0- or 1-word sentences, because the end-of-sentence tokens are understandably one of the more prevalent word types in the corpora. When only the straight count affects the probabilities, this end token will be randomly chosen with more frequency. Also, when the unigram-generated sentences do make it longer than a few tokens, they are much more disjointed, as consecutive words appear that do not agree on tense, gender, plurality, etc. In bigram-generated sentences, this does not happen as much, because conesecutive words have to have appeared in the corpus in order to have any probability of being in a randomly generated sentence. This results in much more logical and meaningful sentences. Also, there are no periods in the middle of bigram-generated sentences, because all periods in the corpus are followed by end tokens. Additionally, there are no instances of several consecutive punctuation marks, because these consecutive pairs also do not appear in the corpora, whereas these do appear in unigram-generated sentences.
