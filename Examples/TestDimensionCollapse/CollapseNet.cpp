//
// Created by Hui Xie on 8/17/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "CollapseNet.h"
#include "LossConvexExample1.h"

CollapseNet::CollapseNet(const string& name, const string& saveDir): FeedForwardNet(name, saveDir){

}
CollapseNet::~CollapseNet(){

}

void CollapseNet::build(){
    // build network
    int id =1;
    InputLayer* inputLayer = new InputLayer(id++, "InputLayer", {3,20,4,12});  //output 3,20,4,12
    addLayer(inputLayer);

    ConvolutionLayer* conv1 = new ConvolutionLayer(id++, "Conv1", getFinalLayer(),{3,3,4,5}, {1,1,1,1}, 13); //output 13*18*8
    addLayer(conv1);
    NormalizationLayer* norm1 = new NormalizationLayer(id++, "Norm1",getFinalLayer(),getFinalLayer()->m_tensorSize);
    addLayer(norm1);
    ReLU* reLU1 = new ReLU(id++, "ReLU1", getFinalLayer(),getFinalLayer()->m_tensorSize);
    addLayer(reLU1);

    ConvolutionLayer* conv2 = new ConvolutionLayer(id++, "Conv2", getFinalLayer(),{5,5,8}, {1,1,1}); //output 9*14
    addLayer(conv2);
    NormalizationLayer* norm2 = new NormalizationLayer(id++, "Norm2",getFinalLayer(),getFinalLayer()->m_tensorSize);
    addLayer(norm2);
    ReLU* reLU2 = new ReLU(id++, "ReLU2", getFinalLayer(),getFinalLayer()->m_tensorSize);
    addLayer(reLU2);

    VectorizationLayer* vec1 = new VectorizationLayer(id++, "Vec1", getFinalLayer());
    addLayer(vec1);
    FCLayer* fc1 = new FCLayer(id++,"fc1", getFinalLayer(), 25);
    addLayer(fc1);
    LossConvexExample1* loss = new LossConvexExample1(id++, "Loss", getFinalLayer());
    addLayer(loss);
}
void CollapseNet::train(){
    InputLayer* inputLayer = getInputLayer();
    Tensor<float> inputTensor(inputLayer->m_pYTensor->getDims());
    LossConvexExample1* lossLayer = (LossConvexExample1*) getFinalLayer();

    int batchSize = getBatchSize();
    int iBatch = 10;

    int i=0;
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
float CollapseNet::test(){
    //null
    return 0;
}
