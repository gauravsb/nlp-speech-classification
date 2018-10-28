
from gensim.models import KeyedVectors
import numpy
import scipy

#from gensim import Word2Vec
from gensim.models import KeyedVectors
#filename = 'GoogleNews-vectors-negative300.bin'
filename = 'glove.840B.300d.w2vformat.txt'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

print("For verb arrive")
result = model.most_similar(positive=['arrive'], topn=10)
print(result)

print("For verb give")
result = model.most_similar(positive=['give'], topn=10)
print(result)

print("For verb enter")
result = model.most_similar(positive=['enter'], topn=10)
print(result)

print("For verb increase")
result = model.most_similar(positive=['increase'], topn=10)
print(result)