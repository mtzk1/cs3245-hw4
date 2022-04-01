import pickle
import os
import math

from TermDictionary import TermDictionary

def SPIMIInvert(tokenStream, outputFile, dictFile):
    """
    This function is akin to the one we've seen the in textbook. Each call to
    SPIMIInvert writes a block to disk.
    """
    tempDict = {} # {term : {docID : [termFreq, weight, vectorLength], docID2 : [termFreq, weight, vectorLength2], ...}, term2 : ...}
    termDict = TermDictionary(dictFile)

    for termDocIDWeightLengthQuartet in tokenStream: # tokenStream is in the form of [(term1, docID, weight, vectorLength), (term2, docID, weight, vectorLength2), ...]
        term = termDocIDWeightLengthQuartet[0]
        docID = termDocIDWeightLengthQuartet[1] 
        weight = termDocIDWeightLengthQuartet[2]
        vectorDocLength = termDocIDWeightLengthQuartet[3] # can have 2 occurence of the same term hence 2 occurence of the vectorLength
        if term not in tempDict:
            tempDict[term] = {}
            tempDict[term][docID] = [1, weight, vectorDocLength]
        else:
            # 2 cases when term is already present in the tempDict:
            #   1. we have seen its docID
            if docID in tempDict[term]:
                tempDict[term][docID][0]+=1

            #   2. we have not seen its docID
            else:
                tempDict[term][docID] = [1, weight, vectorDocLength]

    with open(outputFile, 'wb') as f:
        for term in sorted(tempDict): # {term : {docID : [termFreq, weight, vectorLength], docID2 : [termFreq, weight, vectorLength2], ...}, term2 : ...}
            pointer = f.tell()
            pickle.dump(tempDict[term], f) # store the dictionary {docID : [termFreq, weight, vectorLength], docID2 : [termFreq, weight, vectorLength2], ...}
            termDict.addTerm(term, len(tempDict[term]), pointer) # update TermDictionary
    
    termDict.save()


def retrievePostingsDict(file, pointer):
    """
    Given a pointer to determine the location in disk, 
    retrieves the postings dictionary from that location.
    """
    if pointer == -1: # for non-existent terms
        return {}

    with open(file, 'rb') as f:
        f.seek(pointer)
        postingsList = pickle.load(f)

    return postingsList


def mergeDictsAndPostings(dictFile1, postingsFile1, dictFile2, postingsFile2, outputdictFile, outputPostingsFile):
    """
    This function serves to merge 2 dictionaries and their respective postings files into 1 dictionary and 1 postings file.
    """
    termDict = TermDictionary(outputdictFile) # output dictionary after merging
    dict1 = TermDictionary(dictFile1)
    dict1.load()
    dict2 = TermDictionary(dictFile2)
    dict2.load()
    
    # what I want is to merge 2 dictionaries
    # use term pointers in them to access the postingIDs
    # create a new TermDictionary
    # retrieve the 2 sets of postingsIDs, combine them together
    # get pointer in outputposting file, f.tell()
    # dump the combined postings list into this file
    # update TermDictionary with the term, docFreq (size of set), and pointer
    with open(outputPostingsFile, 'wb') as output:
        keySet1 = set(dict1.getAllKeys()) # all terms only
        keySet2 = set(dict2.getAllKeys()) # all terms only
        unionOfKeys = sorted(keySet1.union(keySet2)) # all (unique) keys (i.e. terms) from the 2 dictionaries to be merged.

        for key in unionOfKeys:
            postings1 = retrievePostingsDict(postingsFile1, dict1.getTermPointer(key)) #retrieves postingsDict if term is present, else {}
            postings2 = retrievePostingsDict(postingsFile2, dict2.getTermPointer(key))
            mergedPostingsDict = mergePostingsDict(postings1, postings2)
            
            pointer = output.tell()
            termDict.addTerm(key, len(mergedPostingsDict), pointer)
            pickle.dump(mergedPostingsDict, output) # storing a dictionary of postings: {docID : termFreq, docID2 : termFreq, ...}

    # end of merging dictionaries and postings file
    # delete the files that have been merged to free up space.
    os.remove(dictFile1)
    os.remove(dictFile2)
    os.remove(postingsFile1)
    os.remove(postingsFile2)
    termDict.save()


def mergePostingsDict(dict1, dict2):
    """
    Merges 2 postings dictionary together.
    Result is in the form of {term1 : [combinedTermFrequency, normWeight], term2 : [combinedTermFrequency, normWeight], ...}
    """
    docIDs1 = set(dict1.keys())
    docIDs2 = set(dict2.keys())
    unionOfDocIDs = sorted(docIDs1.union(docIDs2))
    result = {}

    for docID in unionOfDocIDs:
        result[docID] = [getTermFrequency(dict1, docID) + getTermFrequency(dict2, docID), 
            max(getTermWeight(dict1, docID), getTermWeight(dict2, docID)), max(getVectorDocLength(dict1, docID), getVectorDocLength(dict2, docID))]
            # max() is used because of the way we process files; we process documents fully, and 1024 at a time. Thus, no 2 dictionary file will contain the same
            # docIDs. max() is used because we don't know which of the 2 dictionary contains a particular docID.

    return result
        

def getTermFrequency(postingsDict, docID):
    """
    A clean implementation of retrieving a value from the dictionary.
    Returns the term frequency associated with the key if the key is present.
    Else, return 0.
    """
    try:
        return postingsDict[docID][0]

    except KeyError:
        return 0


def getTermWeight(postingsDict, docID):
    """
    A clean implementation of retrieving a value from the dictionary.
    Returns the weight associated with the key if the key is present.
    Else, return 0.
    """
    try:
        return postingsDict[docID][1]

    except KeyError:
        return 0


def getVectorDocLength(postingsDict, docID):
    """
    A clean implementation of retrieving a value from the dictionary.
    Returns the length of (document) vector associated with the key if the key is present.
    Else, return 0.
    """
    try:
        return postingsDict[docID][2]

    except KeyError:
        return 0


def binaryMerge(dir, fileIDs, outputPostingsFile, outputDictFile):
    """
    Performs binary merge on all files in the specified directory.
    """
    for stage in range(math.ceil(math.log2(fileIDs))): # no. of times we merge is proportional to the no. of unique fileIDs in the directory
        newFileID = 0 # to merged file identifier
        for ID in range(0, fileIDs, 2): # merge files_docID with files_ID+1
            if ID + 1 < fileIDs:
                dictFile1 = dir + 'tempDictionaryFile' + str(ID) + '_stage' + str(stage) + '.txt'
                dictFile2 = dir + 'tempDictionaryFile' + str(ID + 1) + '_stage' + str(stage) + '.txt'
                postingsFile1 = dir + 'tempPostingFile' + str(ID) + '_stage' + str(stage) + '.txt'
                postingsFile2 = dir + 'tempPostingFile' + str(ID + 1) + '_stage' + str(stage) + '.txt'
                outDictFile = dir + 'tempDictionaryFile' + str(newFileID) + '_stage' + str(stage + 1) + '.txt'
                outPostingsFile = dir + 'tempPostingFile' + str(newFileID) + '_stage' + str(stage + 1) + '.txt'
                mergeDictsAndPostings(dictFile1, postingsFile1, dictFile2, postingsFile2, outDictFile, outPostingsFile)
                
            else: # there is an odd number of files in the directory
                oldDictFile = dir + 'tempDictionaryFile' + str(ID) + '_stage' + str(stage) + '.txt'
                newDictFile = dir + 'tempDictionaryFile' + str(newFileID) + '_stage' + str(stage + 1) + '.txt'
                oldPostingsFile = dir + 'tempPostingFile' + str(ID) + '_stage' + str(stage) + '.txt'
                newPostingsFile = dir + 'tempPostingFile' + str(newFileID) + '_stage' + str(stage + 1) + '.txt'
                os.rename(oldDictFile, newDictFile)
                os.rename(oldPostingsFile, newPostingsFile)
            
            newFileID+=1
        fileIDs = newFileID

    # here, i will only have 1 dictionary.txt and 1 postings.txt (the end)
    # move these out into the main directory.
    IDOfLeftoverFiles = newFileID - 1
    StageOfLeftoverFiles = stage + 1
    os.rename(dir + 'tempDictionaryFile' + str(IDOfLeftoverFiles) + '_stage' + str(StageOfLeftoverFiles) + '.txt', outputDictFile)
    os.rename(dir + 'tempPostingFile' + str(IDOfLeftoverFiles) + '_stage' + str(StageOfLeftoverFiles) + '.txt', outputPostingsFile)
