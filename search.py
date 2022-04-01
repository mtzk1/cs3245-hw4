#!/usr/bin/python3
import nltk
import sys
import getopt
import pickle
import math
import heapq

from collections import Counter
from Document import Document
from TermDictionary import TermDictionary


def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")


def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')

    dictFile = TermDictionary(dict_file)
    dictFile.load()  # load term information into dictFile from dict_file

    with open(queries_file, 'r') as queryFile:
        with open(results_file, 'w') as resultFile:
            allResults = []

            for query in queryFile:
                if query.strip():
                    result = cosineScores(query, dictFile, postings_file)
                    allResults.append(result)

                else:
                    allResults.append("")

            outputResult = "\n".join(allResults) # to output all result onto a new line.
            resultFile.write(outputResult)


def retrievePostingsList(file, pointer):
    """
    Given a pointer to determine the location in disk, 
    retrieves the postings list from that location.
    """
    if pointer == -1:  # for non-existent terms
        return []

    with open(file, 'rb') as f:
        f.seek(pointer)
        postingsList = pickle.load(f)

    return postingsList


def cosineScores(query, dictionary, postingsFile):
    """
    Implementation of CosineScore(q) from the textbook.
    """
    stemmer = nltk.stem.porter.PorterStemmer()
    totalNumberOfDocs = len(retrievePostingsList(postingsFile, dictionary.getPointerToDocLengths()))
    result = dict.fromkeys(retrievePostingsList(postingsFile, dictionary.getPointerToDocLengths()).keys(), 0) # in the form of {docID : 1, docID2 : 0.2, ...}

    queryTokens = [stemmer.stem(token.lower()) for token in query.split()]
    qTokenFrequency = Counter(queryTokens) # qTokenFrequency will be in the form of {"the": 2, "and" : 1} if the query is "the and the".
    qToken_tfidfWeights = {term : computeTFIDF(term, frequency, dictionary, totalNumberOfDocs) for term, frequency in qTokenFrequency.items()}
    queryLength = math.sqrt(sum([math.pow(weight, 2) for weight in qToken_tfidfWeights.values()]))
    qTokenNormalisedWeights = {term : normaliseWeight(weight,queryLength) for term, weight in qToken_tfidfWeights.items()}
 
    for term in qTokenNormalisedWeights.keys():
        pointer = dictionary.getTermPointer(term)
        postings = retrievePostingsList(postingsFile, pointer) # in the form of (docID, TermFreq, skipPointer (to be discarded))

        for node in postings:
            docID = node.getDocID()
            termWeight = node.getTermWeight()
            docVectorLength = node.getVectorDocLength()
            result[docID] += normaliseWeight(qTokenNormalisedWeights[term] * termWeight,  docVectorLength) # update with normalised score
    
    # documents and their weights are now settled.

    documentObjects = generateDocumentObjects(result)
    output = extractTop10(documentObjects)

    return " ".join([str(document) for document in output])

def normaliseWeight(weight, vectorLength):
    """
    Given a weight, divide it by the given vectorLength to normalise.
    """
    if vectorLength == 0: # avoids division by 0
        return 0

    else:
        return weight / vectorLength


def computeTFIDF(term, frequency, dictionary, totalNumberOfDocs):
    """
    Takes in a term and computes the tf-idf of a term.
    """
    df = dictionary.getTermDocFrequency(term)
    if (frequency == 0 or df == 0):
        return 0
    else:
        return (1 + math.log10(frequency)) * math.log10(totalNumberOfDocs/dictionary.getTermDocFrequency(term))


def generateDocumentObjects(result):
    """
    Takes in a dictionary of docID-score pairs and create
    a list of Document objects.
    """
    output = []
    for docID, weight in result.items():
        output.append(Document(docID, weight))

    return output

def extractTop10(documentObjects):
    """
    Takes in a list of Document objects and extracts 10 highest scoring documents.
    Less than 10 Document objects will be outputted if there are documents with score = 0
    amongst the supposed 10 highest.
    """

    temp = heapq.nlargest(10, documentObjects) # a list of 10 highest scoring document

    return filter(lambda document : (document.getWeight() > 0), temp)


dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file  = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
