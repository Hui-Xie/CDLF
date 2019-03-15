//
// Created by Hui Xie on 8/16/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.
//

#include "ConvexNet.h"

ConvexNet::ConvexNet(const string& saveDir) : FeedForwardNet(saveDir) {

}

ConvexNet::~ConvexNet() {

}

//Notes: this layerWidthVector does not include LossLayer,  and ReLU and NormalizationLayer do not count as a single layer
// Normalization layer generally put after ReLU
void ConvexNet::build() {
    //build from csv file
    //null
}

void ConvexNet::train() {
    int nIter = 0;
    InputLayer *inputLayer = (InputLayer *) getInputLayer();
    SquareLossLayer *lossLayer = (SquareLossLayer *) getFinalLayer();
    Tensor<float> groundTruth(lossLayer->m_prevLayer->m_tensorSize);
    const int N = groundTruth.getLength();
    for(int i= 0; i<N; ++i){
        groundTruth.e(i) = i;
    }

    int maxIteration = 15000;
    int batchSize = getBatchSize();
    float lr = getLearningRate();
    int numBatch = (maxIteration + batchSize -1)/ batchSize;

    int nBatch = 0;
    while (nBatch < numBatch) {
        if (getJudgeLoss() && lossLayer->getLoss() < getLossTolerance()) {
            break;
        }
        if (isinf(lossLayer->getLoss())) break;

        zeroParaGradient();
        int i = 0;
        for (i = 0; i < batchSize && nIter < maxIteration; ++i) {
            Tensor<float> inputTensor(inputLayer->m_pYTensor->getDims());
            generateGaussian(&inputTensor, 0, 1);
            inputLayer->setInputTensor(inputTensor);
            lossLayer->setGroundTruth(groundTruth);
            forwardPropagate();
            backwardPropagate(true);
            ++nIter;
        }
        sgd(lr, i);
        printIteration(lossLayer, nIter, true);
        ++nBatch;
    }
    lossLayer->printGroundTruth();
}

float ConvexNet::test() {
    //null
    return 0;
}
