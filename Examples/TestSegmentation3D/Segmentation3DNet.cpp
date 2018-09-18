//
// Created by Hui Xie on 9/14/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "Segmentation3DNet.h"



Segmentation3DNet::Segmentation3DNet(const string& name, GNet* pGNet, DNet* pDNet): GAN(name, pGNet, pDNet){

}

Segmentation3DNet::~Segmentation3DNet(){

}






void Segmentation3DNet::trainG(){
   /* InputLayer *inputLayer = getInputLayer();
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) getFinalLayer();

    long maxIteration =420;
    long NTrain = maxIteration;
    int batchSize = getBatchSize();
    float learningRate = getLearningRate();
    long numBatch = maxIteration / batchSize;
    if (0 != maxIteration % batchSize) {
        numBatch += 1;
    }

    long nIter = 0;
    long nBatch = 0;
    //random reshuffle data samples
    vector<long> randSeq = generateRandomSequence(NTrain);
    while (nBatch < numBatch) {
        zeroParaGradient();
        int i = 0;
        for (i = 0; i < batchSize && nIter < maxIteration; ++i) {
            //inputLayer->setInputTensor(m_pMnistData->m_pTrainImages->slice(randSeq[nIter]));
            //lossLayer->setGroundTruth(constructGroundTruth(m_pMnistData->m_pTrainLabels, randSeq[nIter]));
            //forwardPropagate();
            //backwardPropagate();
            ++nIter;
        }
        //sgd(learningRate, i);
        ++nBatch;
    }*/
}

void Segmentation3DNet::trainD(){


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
