import numpy as np
import os
import json
import math
import csv
import collections

### CONSTANTS ###
# SMOOTHING TYPES
SMOOTH_NONE = 0
SMOOTH_POINT1 = 0.1
SMOOTH_POINT3 = 0.3
SMOOTH_ADD1 = 1
SMOOTH_ADD2 = 2
SMOOTH_GT = 3


def read_file(filename):
    stringInFile = ""
    with open(filename, "r", encoding="utf8") as fileObject:
        stringInFile = fileObject.read()
    return stringInFile


def read_file_by_line(filename):
    stringInFile = ""
    with open(filename, encoding="utf8") as f:
        stringInFile += preprocessing(f.readlines())
    return stringInFile

def writeToCSVFile(rowList):
    with open('output.csv', mode='w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Id','Prediction'])
        for string in rowList:
            writer.writerow(string)


def preprocessing(some_text):
    some_text = "<s> " + some_text
    some_text = some_text.replace(" ’", "’")
    some_text = some_text.replace("’ ", "’")
    some_text = some_text.replace("\n", " ")
    some_text = some_text.replace(" . ", " . </s> <s> ")
    some_text = some_text[:-5]  # remove last " <s> "
    return some_text


def token_generation(data):
    tokenList = data.split(" ")
    token_to_id_map = {}
    id_to_token_map = {}
    k = 0
    for i in range(len(tokenList)):
        if tokenList[i] not in token_to_id_map:
            token_to_id_map[tokenList[i]] = k
            id_to_token_map[k] = tokenList[i]
            k += 1
        # id_to_token_map={v: k for k,v in token_to_id_map.items()}

    return token_to_id_map, id_to_token_map, tokenList


def createUnigramMatrix(tokenToIdMap, tokenList, smoothingType):
    numWordTypes = len(tokenToIdMap)

    wordCountMatrix = np.zeros([numWordTypes, 1])

    totalWordLength = len(tokenList)

    # for each word, insert the count into matrix
    for i in range(totalWordLength):
        currentToken = tokenList[i]
        currentTokenId = tokenToIdMap[currentToken]
        wordCountMatrix[currentTokenId][0] += 1

    unsmoothedUnigramCount = wordCountMatrix

    if smoothingType < SMOOTH_GT:  # add-k smoothing
        smoothedUnigramCount = unsmoothedUnigramCount + smoothingType
        smoothedUnigramProb = smoothedUnigramCount / smoothedUnigramCount.sum()
        return smoothedUnigramProb
    elif smoothingType == SMOOTH_GT:  # G-T smoothing
        smoothedUnigramCount = getGtSmoothedNgramCounts(unsmoothedUnigramCount, 1)
        smoothedUnigramProb = smoothedUnigramCount / smoothedUnigramCount.sum()
        return smoothedUnigramProb

    raise Exception('Invalid smoothing type')

# for unigrams with count c on [1, 4], replace all c's in unsmoothed count matrix
# with G-T smoothed count c* = (c + 1) * N_{c+1} / N_c, where N_k = number of
# unigrams with count k.
def getGtSmoothedNgramCounts(unsmoothedCounts, n):
    if (n < 1 or n > 2):
        raise Exception('Invalid value for n')

    counter = np.bincount(unsmoothedCounts.flatten().astype(int))
    smoothedCounts = unsmoothedCounts

    # if unigram, N_0 = 0 --> start loop at 1 to avoid dividing by 0; if bigram, start at 0
    lowestCountToSmooth = 2 - n
    for c in range(lowestCountToSmooth, 5):
        gtCountMatrix = np.full(unsmoothedCounts.shape, (c+1) * counter[c+1] / counter[c])
        smoothedCounts = np.where(unsmoothedCounts == c, gtCountMatrix, smoothedCounts)\

    return smoothedCounts


def createBigramMatrix(tokenToIdMap, tokenList, smoothingType):
    numWordTypes = len(tokenToIdMap)

    # create 2d matrix in numpy
    wordCountMatrix = np.zeros([numWordTypes, numWordTypes])

    totalWordLength = len(tokenList)

    # for each word, insert the count into 2d matrix with
    # for each word occurring together, insert the count into 2d matrix
    for i in range(1, totalWordLength):
        previousToken = tokenList[i - 1]
        currentToken = tokenList[i]
        previousTokenId = tokenToIdMap[previousToken]
        currentTokenId = tokenToIdMap[currentToken]
        wordCountMatrix[previousTokenId][currentTokenId] += 1

    if smoothingType < SMOOTH_GT:  # add-k smoothing
        smoothedBigramCount = wordCountMatrix + smoothingType
        smoothedBigramProb = smoothedBigramCount / smoothedBigramCount.sum(axis=1, keepdims=True)
        return smoothedBigramProb
    elif smoothingType == SMOOTH_GT:
        smoothedBigramCount = getGtSmoothedNgramCounts(wordCountMatrix, 2)
        smoothedBigramProb = smoothedBigramCount / smoothedBigramCount.sum(axis=1, keepdims=True)
        return smoothedBigramProb

    raise Exception('Invalid smoothing type')


def createNGrams(fileName, n, smoothingType):
    # read the file
    text = read_file(fileName)

    # preprocess/cleanup the data
    text = preprocessing(text)

    # generate hash map of [word => index] & [index => word]
    token_to_id_map, id_to_token_map, tokenList = token_generation(text)

    fout = fileName + "tokenToId.txt"
    fo = open(fout, "w", encoding="utf8")

    for k, v in token_to_id_map.items():
        fo.write(str(k) + ' => ' + str(v) + '\n')

    fo.close()

    fout = fileName + "idToToken.txt"
    fo = open(fout, "w", encoding="utf8")

    for k, v in id_to_token_map.items():
        fo.write(str(k) + ' => ' + str(v) + '\n')

    fo.close()

    # print(token_to_id_map)
    # print(id_to_token_map)

    totalWordLength = len(tokenList)

    fo = open('preprocessed.txt', "w+", encoding="utf8")
    fo.write(text)
    fo.close

    if n == 1:
        smoothedUnigramProb = createUnigramMatrix(token_to_id_map, tokenList, smoothingType)
        return smoothedUnigramProb, token_to_id_map, id_to_token_map, totalWordLength
    elif n == 2:
        smoothedBigramProb = createBigramMatrix(token_to_id_map, tokenList, smoothingType)
        return smoothedBigramProb, token_to_id_map, id_to_token_map, totalWordLength
    else:
        raise Exception('Invalid n')


def generateUnigramSentence(unigramProbMatrix, idToTokenMap):
    currentToken = "<s>"
    result = "<s>"
    while (currentToken != "</s>"):
        nextIndex = np.random.choice(np.size(unigramProbMatrix), p=unigramProbMatrix)
        nextToken = idToTokenMap[nextIndex]
        result = result + " " + nextToken
        currentToken = nextToken

    return result


def generateBigramSentence(bigramProbMatrix, tokenToIdMap, idToTokenMap):
    currentToken = "<s>"
    result = "<s>"
    while (currentToken != "</s>"):
        currentIndex = tokenToIdMap[currentToken]
        nextIndex = np.random.choice(np.size(bigramProbMatrix, axis=1),
                                     p=bigramProbMatrix[currentIndex, :])
        nextToken = idToTokenMap[nextIndex]
        result = result + " " + nextToken
        currentToken = nextToken

    return result


def createRandomSentenceObama(unigram, obamaUnigram, obamaBigram, obamaTokenToIdMap, obamaIdTokenMap):
    if unigram:
        print("Creating random sentence from Obama unigram model")
        sentence = generateUnigramSentence(obamaUnigram, obamaIdTokenMap).encode('utf-8')
        fo = open('randomObamaUnigram.txt', "wb")
        fo.write(sentence)
        fo.close
    else:
        print("Creating random sentence from Obama bigram model:")
        sentence = generateBigramSentence(obamaBigram, obamaTokenToIdMap, obamaIdTokenMap).encode('utf-8')
        fo = open('randomObamaBigram.txt', "wb")
        fo.write(sentence)
        fo.close
        # print(generateBigramSentence(obamaBigram, obamaTokenToIdMap, obamaIdTokenMap))


def createRandomSentenceTrump(unigram, trumpUnigram, trumpBigram, trumpTokenToIdMap, trumpIdTokenMap):
    if unigram:
        print("Creating random sentence from Trump unigram model:")
        sentence = generateUnigramSentence(trumpUnigram, trumpIdTokenMap).encode('utf-8')
        fo = open('randomTrumpUnigram.txt', "wb")
        fo.write(sentence)
        fo.close
        # print(generateUnigramSentence(trumpUnigram, trumpIdTokenMap))
    else:
        print("Creating random sentence from Trump bigram model:")
        sentence = generateBigramSentence(trumpBigram, trumpTokenToIdMap, trumpIdTokenMap).encode('utf-8')
        fo = open('randomTrumpBigram.txt', "wb")
        fo.write(sentence)
        fo.close
        # print(generateBigramSentence(trumpBigram, trumpTokenToIdMap, trumpIdTokenMap))


def calculateDevPerplexity(filename, ngramProbMatrix, n, totalTrainingWords, smoothingType):
    rawData = read_file(filename)
    preprocessedData = preprocessing(rawData)
    token_to_id_map, id_to_token_map, tokenList = token_generation(preprocessedData)

    # change smoothing type of 0.3 to "0pt3" to avoid dot in filename
    smoothingTypeStr = "0pt1" if smoothingType == 0.1 else "0pt3" if \
        smoothingType == 0.3 else str(smoothingType)

    if n == 1:
        perplexity = calculateUniPerplexity(ngramProbMatrix, preprocessedData, token_to_id_map, totalTrainingWords, calculateBiPerplexity)

        fout = "output/DevPerplexity_Unigram_k-" + smoothingTypeStr + filename[12:]  # cut off "development/" from filename
        fo = open(fout, "w", encoding="utf8")
        fo.write("Perplexity: " + str(perplexity))
        fo.close()

        return perplexity
    if n == 2:
        perplexity = calculateBiPerplexity(ngramProbMatrix, preprocessedData, token_to_id_map, totalTrainingWords, calculateBiPerplexity)

        fout = "output/DevPerplexity_Bigram_k-" + smoothingTypeStr + filename[12:]  # cut off "development/" from filename
        fo = open(fout, "w", encoding="utf8")
        fo.write("Perplexity: " + str(perplexity))
        fo.close()

        return perplexity



def calculateBiPerplexity(bigramProbMatrix, preprocessedData, token_to_id_map, totalTrainingWords, smoothingType):
    uniqueTokenLength,_ = bigramProbMatrix.shape
    preprocessedData = preprocessedData.split(' ')
    totalWordLength = len(preprocessedData)
    bigramPerplexity = 0.0
    unknownCount = 0

    for i in range(1, len(preprocessedData)):
        previousToken = preprocessedData[i - 1]
        currentToken = preprocessedData[i]
        if (previousToken in token_to_id_map) and (currentToken in token_to_id_map):
            previousTokenId = token_to_id_map[previousToken]
            currentTokenId = token_to_id_map[currentToken]
            #print(math.log(bigramProbMatrix[previousTokenId][currentTokenId]))
            bigramPerplexity += math.log(bigramProbMatrix[previousTokenId][currentTokenId])
        else:
            # Handling for unknown words
            bigramPerplexity += math.log(1 / (totalTrainingWords + uniqueTokenLength))
            #bigramPerplexity += math.log(smoothingType / (totalTrainingWords + smoothingType*uniqueTokenLength))
            #bigramPerplexity += math.log((smoothingType/10)/ (totalTrainingWords + smoothingType*uniqueTokenLength))
            #bigramPerplexity += math.log(smoothingType / (totalTrainingWords))
            unknownCount += 1

    bigramPerplexity = bigramPerplexity * -1
    bigramPerplexity = bigramPerplexity / totalWordLength
    bigramPerplexity = math.exp(bigramPerplexity)

    print("Bigram perplexity: ", bigramPerplexity)

    return bigramPerplexity


def calculateUniPerplexity(unigramProbMatrix, preprocessedData, token_to_id_map, totalTrainingWords, smoothingType):

    uniqueTokenLength = len(unigramProbMatrix)
    print(uniqueTokenLength)
    preprocessedData = preprocessedData.split(' ')
    totalWordLength = len(preprocessedData)
    unigramPerplexity = 0.0
    unknownCount = 0

    for i in range(1, len(preprocessedData)):
        currentToken = preprocessedData[i]
        if (currentToken in token_to_id_map):
            currentTokenId = token_to_id_map[currentToken]
            #print(math.log(bigramProbMatrix[previousTokenId][currentTokenId]))
            unigramPerplexity += math.log(unigramProbMatrix[currentTokenId])
        else:
            # Handline for unknown words
            # make math.log(1 / (prevTokenCount + uniqueTokenLength)) smaller
            # hence, increase: prevTokenCount + uniqueTokenLength
            #unigramPerplexity += math.log(smoothingType / (totalTrainingWords + uniqueTokenLength))
            unigramPerplexity += math.log((smoothingType) / (totalTrainingWords + smoothingType*uniqueTokenLength))
            #unigramPerplexity += math.log(smoothingType / (totalTrainingWords))
            unknownCount += 1

    unigramPerplexity = unigramPerplexity * -1
    unigramPerplexity = unigramPerplexity / totalWordLength
    unigramPerplexity = math.exp(unigramPerplexity)

    print("Unigram perplexity: ", unigramPerplexity)
    #print(unknownCount)

    return unigramPerplexity


def classifyData(fileName, obamaNGram, obamaTokenToIdMap, obamaTotalWords, trumpNGram, trumpTokenToIdMap, trumpTotalWords, n, smoothingType):
    rawData = read_file(fileName)
    rawData = rawData.split('\n')
    rawData = rawData[:-1] # get rid of extra newline
    result = []

    for i in range(len(rawData)):
        preprocessedData = preprocessing(rawData[i])
        #print(i, preprocessedData)
        #print("Test for Obama:")
        if n == 1:
            testObamaPerplexity = calculateUniPerplexity(obamaNGram, preprocessedData, obamaTokenToIdMap, obamaTotalWords, smoothingType)
        else:
            testObamaPerplexity = calculateBiPerplexity(obamaNGram, preprocessedData, obamaTokenToIdMap, obamaTotalWords, smoothingType)

        #print("Test for Trump:")
        if n == 1:
            testTrumpPerplexity = calculateUniPerplexity(trumpNGram, preprocessedData, trumpTokenToIdMap, trumpTotalWords, smoothingType)
        else:
            testTrumpPerplexity = calculateBiPerplexity(trumpNGram, preprocessedData, trumpTokenToIdMap, trumpTotalWords, smoothingType)

        if testObamaPerplexity <= testTrumpPerplexity:
            result.append([i, 0])
        else:
            result.append([i, 1])

    #print(rawData)

    writeToCSVFile(result)


if __name__ == "__main__":
    ############ SET PARAMETERS HERE ############
    # smoothing type constants: SMOOTH_NONE, SMOOTH_POINT1, SMOOTH_POINT3, SMOOTH_ADD1, SMOOTH_ADD2, SMOOTH_GT
    smoothingType = SMOOTH_POINT1
    n = 2

    obamaNGram, obamaTokenToIdMap, obamaIdTokenMap, obamaTotalWords = createNGrams("train/obama.txt", n, smoothingType)
    # createRandomSentenceObama(unigram=True, obamaUnigram, obamaBigram, obamaTokenToIdMap, obamaIdTokenMap)
    # createRandomSentenceObama(unigram=False, obamaUnigram, obamaBigram, obamaTokenToIdMap, obamaIdTokenMap)

    trumpNGram, trumpTokenToIdMap, trumpIdTokenMap, trumpTotalWords = createNGrams("train/trump.txt", n, smoothingType)
    # createRandomSentenceTrump(unigram=True, trumpUnigram, trumpBigram, trumpTokenToIdMap, trumpIdTokenMap)
    # createRandomSentenceTrump(unigram=False, trumpUnigram, trumpBigram, trumpTokenToIdMap, trumpIdTokenMap)

    calculateDevPerplexity("development/obama.txt", obamaNGram, n, obamaTotalWords, smoothingType)
    calculateDevPerplexity("development/trump.txt", trumpNGram, n, trumpTotalWords, smoothingType)

    classifyData("test/test.txt", obamaNGram, obamaTokenToIdMap, obamaTotalWords, trumpNGram, trumpTokenToIdMap, trumpTotalWords, n, smoothingType)
