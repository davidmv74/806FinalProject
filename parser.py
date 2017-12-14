import os
import sys
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import datetime
import pdb
import gzip
import numpy as np
import random

def get_training_vectors(idToQuestions, embeddings):
    training_data = []
    embed_size = len(embeddings["dog"])
    with open('train_random.txt', 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        lines = lines[:1000]
        for line in tqdm(lines):
            splitLine = line.split("\t")
            negatives = splitLine[2][:-1].split(" ")
            totalLines = []
            positives = splitLine[1].split(" ")
            for j in range(len(positives)):
                embeddingsList = []
                random.shuffle(negatives)
                totalLines = [splitLine[0]]+[positives[j]]+negatives[:20]
                for qId in totalLines:
                    embeddingsList.append(get_question_embedding(qId, idToQuestions, embeddings, embed_size))
                training_data.append(embeddingsList)
    return training_data

def get_development_vectors(idToQuestions, embeddings, size):
    dev_data = []
    with open('dev.txt', 'rb') as f:
        lines = f.readlines()
        random.shuffle(lines)
        lines = lines[:size]
        for line in tqdm(lines):
            lineEmbeddings = {}
            splitLines = line.split("\t")
            refQ = Variable(torch.FloatTensor(get_question_embedding(splitLines[0], idToQuestions, embeddings)))
            lineEmbeddings['refq'] = refQ
            candidateEmbeddings = []
            lineEmbeddings['positives'] = splitLines[1]
            for i in range(20):
                candidateId = splitLines[2].split(" ")[i]
                candidateEmbeddings.append((candidateId,Variable(torch.FloatTensor(get_question_embedding(candidateId, idToQuestions, embeddings)))))
            lineEmbeddings['candidates'] =candidateEmbeddings
            dev_data.append(lineEmbeddings)
    return dev_data

def get_domain_train_vectors(setOneQ, setTwoQ, embeddings, size):
    embed_size = len(embeddings["dog"])
    setOneQList = list(setOneQ.keys())
    setTwoQList = list(setTwoQ.keys())
    random.shuffle(setOneQList)
    random.shuffle(setTwoQList)
    setOneQ = [(get_question_embedding(x,setOneQ, embeddings, embed_size) , 0) for x in setOneQList[:size/2]]
    setTwoQ = [(get_question_embedding(x,setTwoQ, embeddings, embed_size) , 0) for x in setTwoQList[:size/2]]
    pairs = list(setOneQ) + list(setTwoQ)
    random.shuffle(pairs)
    pairs = pairs[:size]
    return pairs

def get_questions_id(filee):
    idToQuestions = {}
    with gzip.open(filee, 'rb') as f:
        for file_content in f.readlines():
            qId, qTitle, qBody = file_content.split("\t")
            idToQuestions[qId] = (qTitle.lower(), qBody.lower())
    return idToQuestions

def get_embeddings(filee, word_list):
    embeddings = {}
    if '.gz' in filee:
        f = gzip.open(filee, 'rb')
    else:
        f = open(filee, 'rb')
    f = f.readlines()
    for file_content in tqdm(f):
        word, embedding = file_content.split(" ")[0], file_content.split(" ")[1:-1]
        if word in word_list:
            embeddings[word] = [float(emb) for emb in embedding]
    return embeddings

def get_question_embedding(qId, idToQuestions, embeddings, embedding_size):
    title, body  = idToQuestions[qId]
    bodyEmbedding = np.zeros(embedding_size)
    for word in body.split(" "):
        if word in embeddings:
            bodyEmbedding += embeddings[word]
    bodyEmbedding /= len(body)

    titleEmbedding = np.zeros(embedding_size)
    for word in title.split(" "):
        if word in embeddings:
            titleEmbedding += embeddings[word]
    titleEmbedding /= len(title)

    return (titleEmbedding+bodyEmbedding)/2

def get_android_samples(negFile, posFile, embeddings, idToQuestions, size):
    posPairs = {}
    negPairs = {}
    embed_size = len(embeddings["dog"])
    with open(negFile, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            ref, neg = line.split(" ")
            if ref in negPairs:
                negPairs[ref].append(neg[:-1])
            else:
                negPairs[ref] = [neg[:-1]]
    with open(posFile, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            ref, pos = line.split(" ")
            if ref in posPairs:
                posPairs[ref].append(pos[:-1])
            else:
                posPairs[ref] = [pos[:-1]]
    posPairsList = posPairs.items()
    random.shuffle(posPairsList)
    posPairsList = posPairsList[:size]
    samples = []
    for qPair in posPairsList:
        target = []
        random.shuffle(qPair[1])
        posCands = qPair[1][:40]
        random.shuffle(negPairs[qPair[0]])
        negCands = negPairs[qPair[0]][:max(len(posCands), 10)]
        posCands = [(get_question_embedding(x, idToQuestions, embeddings, embed_size), 1) for x in posCands]
        negCands = [(get_question_embedding(x, idToQuestions, embeddings, embed_size), 0) for x in negCands]
        cands = posCands+negCands
        random.shuffle(cands)
        cands = cands[:20]
        samples.append(cands)
    return samples
