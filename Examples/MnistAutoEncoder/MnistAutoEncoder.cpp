//
// Created by Hui Xie on 12/11/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "MnistAutoEncoder.h"

MnistAutoEncoder::MnistAutoEncoder(const string &name, const string &saveDir,  MNIST* pMnistData) : FeedForwardNet(name, saveDir) {
  m_pMnistData = pMnistData;
}

MnistAutoEncoder::~MnistAutoEncoder() {

}


void MnistAutoEncoder::build() {
   //null
}


void MnistAutoEncoder::train() {
    InputLayer *inputLayer = getInputLayer();
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) getFinalLayer();

    long maxIteration =m_pMnistData->m_pTrainLabels->getLength();
    long NTrain = maxIteration;
    int batchSize = getBatchSize();
    float learningRate = getLearningRate();
    long numBatch = (maxIteration + batchSize -1) / batchSize;
    long nIter = 0;
    long nBatch = 0;
    //random reshuffle data samples
    vector<long> randSeq = generateRandomSequence(NTrain);
    while (nBatch < numBatch) {
        zeroParaGradient();
        int i = 0;
        for (i = 0; i < batchSize && nIter < maxIteration; ++i) {
            inputLayer->setInputTensor(m_pMnistData->m_pTrainImages->slice(randSeq[nIter]));
            lossLayer->setGroundTruth(constructGroundTruth(m_pMnistData->m_pTrainLabels, randSeq[nIter]));
            forwardPropagate();
            backwardPropagate(true);
            ++nIter;
        }
        sgd(learningRate, i);
        ++nBatch;
    }
}



float MnistAutoEncoder::test(){
    InputLayer *inputLayer = getInputLayer();
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) getFinalLayer();
    long n = 0;
    long nSuccess = 0;
    const long Ntest = m_pMnistData->m_pTestLabels->getLength();
    while (n < Ntest) {
        inputLayer->setInputTensor(m_pMnistData->m_pTestImages->slice(n));
        lossLayer->setGroundTruth(constructGroundTruth(m_pMnistData->m_pTestLabels, n));
        forwardPropagate();
        if (lossLayer->predictSuccessInColVec()) ++nSuccess;
        ++n;
    }
    cout<<"Info: nSuccess = "<<nSuccess<<" in "<<Ntest<<" test samples."<<endl;
    return  nSuccess * 1.0 / Ntest;
}

//construct a 2*1 one-hot vector
Tensor<float> MnistAutoEncoder::constructGroundTruth(Tensor<unsigned char> *pLabels, const long index) {
    Tensor<float> tensor({10, 1});
    tensor.zeroInitialize();
    tensor.e(pLabels->e(index)) = 1;
    return tensor;
}
