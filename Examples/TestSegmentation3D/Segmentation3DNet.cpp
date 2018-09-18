//
// Created by Hui Xie on 9/14/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "Segmentation3DNet.h"



Segmentation3DNet::Segmentation3DNet(const string& name, GNet* pGNet, DNet* pDNet): GAN(name, pGNet, pDNet){

}

Segmentation3DNet::~Segmentation3DNet(){

}

void Segmentation3DNet::trainG(const int N){
    InputLayer* inputLayer = m_pGNet->m_pInputXLayer;
    CrossEntropyLoss* lossGxLayer = (CrossEntropyLoss *) m_pGNet->getFinalLayer();
    CrossEntropyLoss* lossDLayer = (CrossEntropyLoss *) m_pDNet->getFinalLayer();

    long maxIteration = N;
    long NTrain = maxIteration;
    int batchSize = m_pGNet->getBatchSize();
    float learningRateG = m_pGNet->getLearningRate();
    long numBatch = maxIteration / batchSize;
    if (0 != maxIteration % batchSize) {
        numBatch += 1;
    }

    long nIter = 0;
    long nBatch = 0;
    //random reshuffle data samples
    vector<long> randSeq = generateRandomSequence(NTrain);
    while (nBatch < numBatch) {
        m_pGNet->zeroParaGradient();
        m_pDNet->zeroParaGradient();
        int i = 0;
        for (i = 0; i < batchSize && nIter < maxIteration; ++i) {
            //todo: set input and groundtruth
            //inputLayer->setInputTensor();
            //lossGxLayer->setGroundTruth(constructGroundTruth(m_pMnistData->m_pTrainLabels, randSeq[nIter]));
            //lossDLayer->setGroundTruth();
            forwardG();
            backwardG();
            ++nIter;
        }
        m_pGNet->sgd(learningRateG, i);
        ++nBatch;
    }
}

void Segmentation3DNet::trainD(const int N){
    InputLayer* inputLayer = m_pDNet->m_pInputXLayer;
    Layer* GxLayer = m_pDNet->m_pGxLayer;
    Layer* GTLayer = m_pDNet->m_pGTLayer;
    CrossEntropyLoss* lossDLayer = (CrossEntropyLoss *) m_pDNet->getFinalLayer();

    long maxIteration = N;
    long NTrain = maxIteration;
    int batchSize = m_pGNet->getBatchSize();
    float learningRateG = m_pGNet->getLearningRate();
    long numBatch = maxIteration / batchSize;
    if (0 != maxIteration % batchSize) {
        numBatch += 1;
    }

    long nIter = 0;
    long nBatch = 0;
    //random reshuffle data samples
    vector<long> randSeq = generateRandomSequence(NTrain);
    while (nBatch < numBatch) {
        m_pDNet->zeroParaGradient();
        int i = 0;
        for (i = 0; i < batchSize && nIter < maxIteration; ++i) {
            switchDToGx();
            //todo: set Gx and loss Label
            //inputLayer->setInputTensor();
            //lossGxLayer->setGroundTruth(constructGroundTruth(m_pMnistData->m_pTrainLabels, randSeq[nIter]));
            //lossDLayer->setGroundTruth();
            forwardD();
            backwardD();

            switchDToGT();
            //todo: set GT and  loss label
            forwardD();
            backwardD();
            ++nIter;
        }
        m_pDNet->sgd(learningRateG, i);
        ++nBatch;
    }

}


float Segmentation3DNet::test(){
   /* InputLayer *inputLayer = getInputLayer();
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) getFinalLayer();
    long n = 0;
    long nSuccess = 0;
    const long Ntest= 10000; //= m_pMnistData->m_pTestLabels->getLength();
    while (n < Ntest) {
        //inputLayer->setInputTensor(m_pMnistData->m_pTestImages->slice(n));
        //lossLayer->setGroundTruth(constructGroundTruth(m_pMnistData->m_pTestLabels, n));
        //forwardPropagate();
        if (lossLayer->predictSuccessInColVec()) ++nSuccess;
        ++n;
    }
    cout<<"Info: nSuccess = "<<nSuccess<<" in "<<Ntest<<" test samples."<<endl;
    return  nSuccess * 1.0 / Ntest;*/
}

//construct a 2*1 one-hot vector
Tensor<float> Segmentation3DNet::constructGroundTruth(Tensor<unsigned char> *pLabels, const long index) {
    Tensor<float> tensor({10, 1});
    tensor.zeroInitialize();
    tensor.e(pLabels->e(index)) = 1;
    return tensor;
}
