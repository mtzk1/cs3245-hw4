#!/usr/bin/python3
import shutil
import nltk
import sys
import getopt
import os
import pickle
import math
import csv

from TermDictionary import TermDictionary
from Node import Node
from SPIMI import SPIMIInvert, binaryMerge


maxs = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxs)
        break
    except OverflowError:
        maxs = int(maxs/10)


def usage():
    print("usage: " + sys.argv[0] + " -i input-file -d dictionary-file -p postings-file")


def build_index(in_file, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print('indexing...')

    tempFile = 'temp.txt'
    workingDirectory = "workingDirectory/"
    limit = 1024  # max number of docs to be processed at any 1 time.
    result = TermDictionary(out_dict)

    file = open(in_file, 'r', encoding="utf8")
    csvreader = csv.reader(file)
    fields = next(csvreader)

    # set up temp directory for SPIMI process
    if not os.path.exists(workingDirectory):
        os.mkdir(workingDirectory)
    else:
        shutil.rmtree(workingDirectory)  # delete the specified directory tree for re-indexing purposes
        os.mkdir(workingDirectory)

    fileID = 0
    stageOfMerge = 0
    count = 0
    tokenStream = []
    sortedDocIDs = []
    docLengths = {}  # {docID : length, docID2 : length, ...}, to be added dumped into the postings file with its pointer stored in the final termDictionary file

    for row in csvreader:
        sortedDocIDs.append(row[0])
        result = generateTokenStreamWithVectorLength(row[0], row[2])  # returns an array of terms present in that particular doc
        tokenStream.extend(result[0])
        docLengths[row[0]] = result[1]
        count += 1

        if count == limit:  # no. of docs == limit
            outputPostingsFile = workingDirectory + 'tempPostingFile' + str(fileID) + '_stage' + str(stageOfMerge) + '.txt'
            outputDictionaryFile = workingDirectory + 'tempDictionaryFile' + str(fileID) + '_stage' + str(stageOfMerge) + '.txt'
            SPIMIInvert(tokenStream, outputPostingsFile, outputDictionaryFile)
            fileID += 1
            count = 0  # reset counter
            tokenStream = []  # clear tokenStream
    
    if count > 0:  # in case the number of files isnt a multiple of the limit set
        outputPostingsFile = workingDirectory + 'tempPostingFile' + str(fileID) + '_stage' + str(stageOfMerge) + '.txt'
        outputDictionaryFile = workingDirectory + 'tempDictionaryFile' + str(fileID) + '_stage' + str(stageOfMerge) + '.txt'
        SPIMIInvert(tokenStream, outputPostingsFile, outputDictionaryFile)
        fileID += 1  # passed into binary merge, and it will be for i in range(0, fileID, 2) --> will cover everything

    file.close()

    # inverting done. Tons of dict files and postings files to merge
    binaryMerge(workingDirectory, fileID, tempFile, out_dict)
    result = TermDictionary(out_dict)
    result.load()

    convertToPostingNodes(out_postings, tempFile, result)
    
    # add docLengths into output postings file, and store a pointer in the resultant dictionary.
    with open(out_postings, 'ab') as f:  # append to postings file
        pointer = f.tell()
        result.addPointerToDocLengths(pointer)
        pickle.dump(docLengths, f)

    result.save()

    os.remove(tempFile)
    shutil.rmtree(workingDirectory, ignore_errors=True)


def generateTokenStreamWithVectorLength(docID, content):
    """
    Given a document and the directory, return a tuple of 2 items: first is a list of (term, docID, weight, lengthofDocVector).
    Second is the length of the document. 
    We apply case-folding + stemming to all tokens encountered.
    Weight of a term simply 1 + log10(termFrequency), with no idf component.
    """
    stemmer = nltk.stem.porter.PorterStemmer()

    length = 0
    countOfTerms = {}  # will be in the form of {term1 : count, term2 : count, ...}

    sentences = nltk.tokenize.sent_tokenize(content)
    for sentence in sentences:
        words = nltk.tokenize.word_tokenize(sentence)
        for word in words:
            length += 1
            stemmedWord = stemmer.stem(word.lower())  # stemming + case-folding

            if stemmedWord in countOfTerms:
                countOfTerms[stemmedWord] += 1

            else:
                countOfTerms[stemmedWord] = 1

    weightOfTerms = {term: 1 + math.log10(value) for term, value in countOfTerms.items()}  # no idf
    lengthOfDocVector = math.sqrt(sum([count**2 for count in weightOfTerms.values()]))

    output = [(term, docID, weight, lengthOfDocVector) for term, weight in weightOfTerms.items()]  # all terms in a particular document, and its associated term weight, and length of vector

    return output, length  # returns a tuple: (a list of processed terms in the form of  [(term1, docID, weight, docVectorLength), (term2, docID, docVectorLength), ...], length of document)


def convertToPostingNodes(out_postings, file, termDictionary):
    """
    We convert all postings in the postings file into Node objects,
    where each Node object stores a docID, the term frequency in document <docID>, 
    the term weight, and the vector length of document <docID>.
    These Node objects are saved into out_postings.
    """
    with open(file, 'rb') as ref:
        with open(out_postings, 'wb') as output:

            termDict = termDictionary.getTermDict()
            for term in termDict:
                pointer = termDict[term][1]  # retrieves pointer associated to the term
                ref.seek(pointer)
                docIDsDict = pickle.load(ref)  # loads a dictionary of docIDs

                postingsNodes = [Node(docID, docIDsDict[docID][0], docIDsDict[docID][1], docIDsDict[docID][2]) for docID in docIDsDict] # create Nodes
                newPointer = output.tell()  # new pointer location
                pickle.dump(postingsNodes, output)
                termDictionary.updatePointerToPostings(term, newPointer)  # term entry is now --> term : [docFreq, pointer]


input_file = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input file
        input_file = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_file == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_file, output_file_dictionary, output_file_postings)
