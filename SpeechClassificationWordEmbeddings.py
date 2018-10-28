

from gensim.models import KeyedVectors
import numpy as np
import csv

def read_file(filename):
    stringInFile = ""
    with open(filename, "r", encoding="utf8") as fileObject:
        stringInFile = fileObject.read()
    return stringInFile


def preprocessing(some_text):
    #some_text = some_text.replace("\n", " ")
    some_text = some_text.replace("\t", " ")
    some_text = some_text.replace("  ", " ")
    some_text = some_text.replace(" ’", "’")
    some_text = some_text.replace("’ ", "’")
    some_text = some_text.replace("’  ", "’")
    some_text = some_text.replace("  ’", "’")
    return some_text


def wordEmbeddingCalculation(model, tokenList):
    k = 0
    while True:
        try:
            if tokenList[k]:
                total = model.word_vec(tokenList[k])
                break
        except KeyError:
            k = k+1
            continue

    counter = 1
    for i in range(k+1, len(tokenList)):
            if tokenList[i]:
                try:
                    vec = model.word_vec(tokenList[i])
                except KeyError:
                    continue
                total = np.add(vec, total)
                counter += 1

    finalVector = total / counter
    return finalVector

def parse_word_file(filename, model):
    fileText = read_file(filename)
    fileText = preprocessing(fileText)
    lineList = fileText.split('\n')
    lineList = lineList[:-1]
    vectorList = []

    for str in lineList:
        #print(str)
        tokenList = str.split(" ")
        lineVector = wordEmbeddingCalculation(model, tokenList)
        vectorList.append(lineVector)

    return vectorList



def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity according
    to the definition of the dot product
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def writeToCSVFile(rowList):
    with open('output3-glove.csv', mode='w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Id','Prediction'])
        for string in rowList:
            writer.writerow(string)


def classify(fileName, obamaVectorList, trumpVectorList, model):
    rawData = read_file(fileName)
    rawData = rawData.split('\n')
    rawData = rawData[:-1] # get rid of extra newline
    result = []

    for i in range(len(rawData)):
        preprocessedData = preprocessing(rawData[i])
        vectorToCompare = wordEmbeddingCalculation(model, preprocessedData.split(' '))
        obamaMaxCosineSimilarity = -1.5
        trumpMaxCosineSimilarity = -1.5

        for obamaVector in obamaVectorList:
            obamaMaxCosineSimilarity = max(cos_sim(obamaVector, vectorToCompare), obamaMaxCosineSimilarity)

        for trumpVector in trumpVectorList:
            trumpMaxCosineSimilarity = max(cos_sim(trumpVector, vectorToCompare), trumpMaxCosineSimilarity)

        if obamaMaxCosineSimilarity < trumpMaxCosineSimilarity:
            result.append([i, 1])
        else:
            result.append([i, 0])

    writeToCSVFile(result)



if __name__ == "__main__":
    filename = 'glove.840B.300d.w2vformat.txt'
    #filename = 'GoogleNews-vectors-negative300.bin'
    #model = KeyedVectors.load_word2vec_format(filename, binary=True)
    model = KeyedVectors.load_word2vec_format(filename, binary=False)
    obamaVectorList = parse_word_file("train/obama.txt", model)
    trumpVectorList = parse_word_file("train/trump.txt", model)

    classify("test/test.txt", obamaVectorList, trumpVectorList, model)

