from gensim.models import KeyedVectors
import numpy
import scipy
import csv

'''
#from gensim import Word2Vec
from gensim.models import KeyedVectors
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
print("Harshit")
m = model.word_vec("boy")
print(m)
print (type(m))
#result = model.most_similar(positive=['girl', 'brother'], negative=['boy'], topn=1)
#print(result)
'''


def read_file(filename):
    stringInFile = ""
    with open(filename, "r", encoding="utf8") as fileObject:
        stringInFile = fileObject.read()
    return stringInFile


def preprocessing(some_text):
    some_text = some_text.replace("\n", " ")
    some_text = some_text.replace("\t", " ")
    some_text = some_text.replace("  ", " ")
    return some_text


def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity according
    to the definition of the dot product
    """
    dot_product = numpy.dot(a, b)
    norm_a = numpy.linalg.norm(a)
    norm_b = numpy.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def process(filename, token_to_vector_map):
    stringInFile = ""
    with open(filename, encoding="utf8") as f:
        # stringInFile += preprocessing(f.readlines())
        count = 0
        precision = 0.0
        for str in f.readlines():
            count = count + 1
            str = preprocessing(str)
            rowWordList = str.split(" ")
            first = rowWordList[0]
            second = rowWordList[1]
            third = rowWordList[2]
            fourth = rowWordList[3]
            # print (first, second, third, fourth)
            # print (token_to_vector_map.get(second).shape)
            # print (token_to_vector_map.get(third).shape)
            # print (token_to_vector_map.get(second))
            temp = numpy.add(token_to_vector_map.get(second), token_to_vector_map.get(third))
            temp = numpy.subtract(temp, token_to_vector_map.get(first))
            max = -1.5
            # expected = cos_sim(temp, token_to_vector_map.get(rowWordList[3]))
            # print ("Expected cosine similarity ", expected)
            wordWithMaxCosineSimilarity = ""
            for k, v in token_to_vector_map.items():
                # cosineSimilarity = 1-scipy.spatial.distance.cosine(temp,token_to_vector_map.get(k))
                if (k == first or k == second or k == third):
                    continue
                cosineSimilarity = cos_sim(temp, token_to_vector_map.get(k))
                # print ("Cosine similarity with " + uniqueWord + " to compare to " + fourth)
                # print (cosineSimilarity)
                if (cosineSimilarity > max):
                    max = cosineSimilarity
                    wordWithMaxCosineSimilarity = k
            #print (wordWithMaxCosineSimilarity, max)
            if (wordWithMaxCosineSimilarity == fourth):
                precision = precision + 1

        print (precision)
        print (count)
        print (precision / count)
    return stringInFile


def parse_word_file(filename):
    fileText = read_file(filename)
    fileText = preprocessing(fileText)
    tokenList = fileText.split(" ")
    token_to_vector_map = {}
    filename = 'GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(filename, binary=True)

    for i in range(len(tokenList)):
        if tokenList[i] not in token_to_vector_map:
            if tokenList[i]:
                # print (tokenList[i])
                token_to_vector_map[tokenList[i]] = model.word_vec(tokenList[i])

    return token_to_vector_map


'''
    fout = "wordWithVector.txt"
    fo = open(fout, "w", encoding="utf8")

    for k, v in token_to_vector_map.items():
            fo.write(str(k) + ' ' + str(v) + '\n')

    fo.close()
'''

if __name__ == "__main__":
    token_to_vector_map = parse_word_file("analogy_test.txt")
    process("analogy_test.txt", token_to_vector_map)
