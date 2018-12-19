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
    MeanSquareLossLayer *lossLayer = (MeanSquareLossLayer *) getFinalLayer();

    const int maxIteration =m_pMnistData->m_pTrainLabels->getLength();
    const int NTrain = maxIteration;
    const int batchSize = getBatchSize();
    const float learningRate = getLearningRate();
    const int numBatch = (maxIteration + batchSize -1) / batchSize;
    int nIter = 0;
    int nBatch = 0;
    //random reshuffle data samples
    vector<int> randSeq = generateRandomSequence(NTrain);
    while (nBatch < numBatch) {
        zeroParaGradient();
        int i = 0;
        for (i = 0; i < batchSize && nIter < maxIteration; ++i) {
            Tensor<unsigned  char> inputImage = m_pMnistData->m_pTrainImages->slice(randSeq[nIter]);
            inputLayer->setInputTensor(inputImage);
            lossLayer->setGroundTruth(inputImage);
            forwardPropagate();
            backwardPropagate(true);
            ++nIter;
        }
        sgd(learningRate, i);
        ++nBatch;

        //debug
        //if (nBatch >= numBatch){
        //    cout<<"Information: the final batchSize is "<< i<<endl;
        //    lossLayer->m_pGroundTruth->save(m_directory +"/" + "GroundTruth.csv");
        //}
    }
}


float MnistAutoEncoder::test(){
    InputLayer *inputLayer = getInputLayer();
    MeanSquareLossLayer *lossLayer = (MeanSquareLossLayer *) getFinalLayer();
    int n = 0;
    const int Ntest = m_pMnistData->m_pTestLabels->getLength();
    float squareLoss = 0.0;
    while (n < Ntest) {
        Tensor<unsigned  char> inputImage = m_pMnistData->m_pTrainImages->slice(n);
        inputLayer->setInputTensor(inputImage);
        lossLayer->setGroundTruth(inputImage);
        forwardPropagate();
        squareLoss += lossLayer->getLoss();
        ++n;
    }
    return  squareLoss / Ntest;
}

void MnistAutoEncoder::autoEncode(const Tensor<float>& inputImage, int& predictLabel, Tensor<float>& reconstructImage){
    InputLayer *inputLayer = getInputLayer();
    FCLayer* predictLayer = (FCLayer*)getLayer(18); // the FC2 layer.
    MeanSquareLossLayer *lossLayer = (MeanSquareLossLayer *) getFinalLayer();
    inputLayer->setInputTensor(inputImage);
    lossLayer->setGroundTruth(inputImage);
    forwardPropagate();
    predictLabel = predictLayer->m_pYTensor->maxPosition();
    reconstructImage = *lossLayer->m_prevLayer->m_pYTensor;
}
