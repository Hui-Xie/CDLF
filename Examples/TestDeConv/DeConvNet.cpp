
//
// Created by Hui Xie on 01/15/19.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#include "DeConvNet.h"


DeConvNet::DeConvNet(const string &saveDir) : FeedForwardNet(saveDir) {

}

DeConvNet::~DeConvNet() {
    //null
}

void DeConvNet::build() {
    //null: use csv file to create network
}

void DeConvNet::train() {
    InputLayer *inputLayer = getInputLayer();
    SquareLossLayer *lossLayer = (SquareLossLayer *) getFinalLayer();

    vector<int> inputSize = inputLayer->m_tensorSize;
    Tensor<float> inputTensor(inputSize);
    if(m_OneSampleTrain){
        const int length = inputTensor.getLength();
        for (int i=0; i< length; ++i){
            inputTensor.e(i) = i/10.0;
        }
        inputLayer->setInputTensor(inputTensor);
    }


    vector<int> outputSize = lossLayer->m_prevLayer->m_tensorSize;
    Tensor<float> groundTruthTensor(outputSize);
    for(int i =0; i<groundTruthTensor.getLength(); ++i){
        groundTruthTensor.e(i) = i;
    }
    lossLayer->setGroundTruth(groundTruthTensor);


    int N = 100;
    if (m_OneSampleTrain){
        N = 1;
    }
    int batchSize = getBatchSize();
    const float learningRate = getLearningRate();
    const int numBatch = (N + batchSize -1) / batchSize;
    int nIter = 0;
    int nBatch = 0;
    while (nBatch < numBatch) {
        zeroParaGradient();
        int i = 0;
        for (i = 0; i < batchSize && nIter < N; ++i) {
            if (!m_OneSampleTrain){
                generateGaussian(&inputTensor, 0, 1);
                inputLayer->setInputTensor(inputTensor);
            }

            forwardPropagate();
            backwardPropagate(true);
            ++nIter;
        }
        averageParaGradient(i);
        optimize("sgd");
        ++nBatch;

        //savedYTensor();
    }

    m_loss = lossLayer->getLoss();

/*
    //debug
    TransposedConvolutionLayer * tConv = (TransposedConvolutionLayer *) getLayer(10);
    ConvolutionLayer* conv = (ConvolutionLayer*) getLayer(20);

    cout<<"TConv dW : "<<endl;
    for (int i= 0; i< tConv->m_numFilters;++i){
        tConv->m_pdW[i]->print();
    }
    cout<<"TConv dB : "<<endl;
    tConv->m_pdB->print();

    cout<<"TConv dY.sum = "<<tConv->m_pdYTensor->sum() <<endl;
    cout<<"==============================="<<endl;

    cout<<"Conv dW : "<<endl;
    for (int i= 0; i< conv->m_numFilters;++i){
        conv->m_pdW[i]->print();
    }
    cout<<"conv dB : "<<endl;
    conv->m_pdB->print();
    cout<<"conv dY.sum = "<<conv->m_pdYTensor->sum() <<endl;
    cout<<"==============================="<<endl;

    printLayersDY();

    cout<<"==============================="<<endl;
    cout<<"==============================="<<endl;


    printLayersY();
*/



}

float DeConvNet::test() {
    InputLayer *inputLayer = getInputLayer();
    SquareLossLayer *lossLayer = (SquareLossLayer *) getFinalLayer();

    vector<int> outputSize = lossLayer->m_prevLayer->m_tensorSize;
    Tensor<float> groundTruthTensor(outputSize);
    for (int i = 0; i < groundTruthTensor.getLength(); ++i) {
        groundTruthTensor.e(i) = i;
    }
    lossLayer->setGroundTruth(groundTruthTensor);

    m_loss = 0;
    const int N = 20;
    int nIter = 0;
    while (nIter < N) {
        vector<int> inputSize = inputLayer->m_tensorSize;
        Tensor<float> inputTensor(inputSize);
        generateGaussian(&inputTensor, 0, 1);
        inputLayer->setInputTensor(inputTensor);

        forwardPropagate();
        m_loss += lossLayer->getLoss();
        ++nIter;
    }
    //printf("output tensor as example:\n");
    //lossLayer->m_prevLayer->m_pYTensor->print();
    m_loss /= N;
    return m_loss;
}

