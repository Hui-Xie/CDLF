//
// Created by Hui Xie on 8/16/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "ConvexNet.h"

ConvexNet::ConvexNet(const string &name, const string& saveDir, const vector<int> &layerWidthVector) : FeedForwardNet(name, saveDir) {
    m_layerWidthVector = layerWidthVector;
}

ConvexNet::~ConvexNet() {

}

//Notes: this layerWidthVector does not include LossLayer,  and ReLU and NormalizationLayer do not count as a single layer
// Normalization layer generally put after ReLU
void ConvexNet::build() {
    int nLayers = m_layerWidthVector.size();
    if (0 == nLayers) {
        cout << "FeedForwardNet has at least one layer." << endl;
        return;
    }
    int layerID = 1;
    InputLayer *inputLayer = new InputLayer(layerID++, "InputLayer", {m_layerWidthVector.at(0), 1});
    addLayer(inputLayer);
    for (int i = 1; i < nLayers; ++i) {
        FCLayer *fcLayer = new FCLayer(layerID++, "FCLayer" + to_string(i), getFinalLayer(),
                                       m_layerWidthVector.at(i));
        addLayer(fcLayer);
        if (i != nLayers - 1) {
            ReLU *reLU = new ReLU(layerID++, "ReLU" + to_string(i), getFinalLayer(),getFinalLayer()->m_tensorSize);
            addLayer(reLU);
            NormalizationLayer *normalLayer = new NormalizationLayer(layerID++, "NormLayer" + to_string(i),
                                                                     getFinalLayer(),getFinalLayer()->m_tensorSize);
            addLayer(normalLayer);
        }
    }
    //convex example 1: f= \sum (x_i-i)^2
    LossConvexExample1 *lossLayer = new LossConvexExample1(100, "ConvexLossLayer", getFinalLayer());
    //convex example 2: f= = \sum exp(x_i -i)
    //LossConvexExample2* lossLayer = new LossConvexExample2(1003, "ConvexLossLayer",  net.getFinalLayer());
    addLayer(lossLayer);
}

void ConvexNet::train() {
    int nIter = 0;
    InputLayer *inputLayer = (InputLayer *) getInputLayer();
    MeanSquareLossLayer *lossLayer = (MeanSquareLossLayer *) getFinalLayer();
    Tensor<float> groundTruth(lossLayer->m_prevLayer->m_pYTensor->getDims());
    const int N = groundTruth.getLength();
    for(int i= 0; i<N; ++i){
        groundTruth.e(i) = i;
    }

    int maxIteration = 1000;
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
