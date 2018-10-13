//
// Created by Hui Xie on 8/17/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "Conv4DNet.h"
#include "LossConvexExample1.h"

Conv4DNet::Conv4DNet(const string& name): FeedForwardNet(name){

}
Conv4DNet::~Conv4DNet(){

}

void Conv4DNet::build(){
    // build network
    int id =1;
    InputLayer* inputLayer = new InputLayer(id++, "InputLayer", {7,7,7,7});  //output 7*7*7*7
    addLayer(inputLayer);

    ConvolutionLayer* conv1 = new ConvolutionLayer(id++, "Conv1", getFinalLayer(), {7,3,3,3}, 1); //output 5*5*5
    addLayer(conv1);
    NormalizationLayer* norm1 = new NormalizationLayer(id++, "Norm1",getFinalLayer());
    addLayer(norm1);
    ReLU* reLU1 = new ReLU(id++, "ReLU1", getFinalLayer());
    addLayer(reLU1);

    ConvolutionLayer* conv2 = new ConvolutionLayer(id++, "Conv2",getFinalLayer(), {5,3,3}, 1); //output 3*3
    addLayer(conv2);

    VectorizationLayer* vec1 = new VectorizationLayer(id++, "Vec1", getFinalLayer());
    addLayer(vec1);
    FCLayer* fc1 = new FCLayer(id++,"fc1", getFinalLayer(), 9);
    addLayer(fc1);
    LossConvexExample1* loss = new LossConvexExample1(id++, "Loss", getFinalLayer());
    addLayer(loss);
}
void Conv4DNet::train(){
    InputLayer* inputLayer = getInputLayer();
    Tensor<float> inputTensor(inputLayer->m_pYTensor->getDims());
    LossConvexExample1* lossLayer = (LossConvexExample1*) getFinalLayer();

    int batchSize = getBatchSize();
    int iBatch = 30;

    long i=0;
    while (i< iBatch){
        zeroParaGradient();
        for(int j=0; j<batchSize; ++j){
            generateGaussian(&inputTensor,0,1);
            inputLayer->setInputTensor(inputTensor);
            forwardPropagate();
            backwardPropagate(true);
        }
        sgd(getLearningRate(), batchSize);
        printIteration(lossLayer, i);
        ++i;
    }
    lossLayer->printGroundTruth();
}
float Conv4DNet::test(){
    //null
    return 0;
}
