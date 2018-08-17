//
// Created by Hui Xie on 8/16/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "ConvNet.h"
#include "LossConvexExample1.h"

ConvNet::ConvNet(){

}
ConvNet::~ConvNet(){

}

void ConvNet::build(){
    // build network
    int id =1;
    InputLayer* inputLayer = new InputLayer(id++, "InputLayer", {7,7});
    addLayer(inputLayer);

    ConvolutionLayer* conv1 = new ConvolutionLayer(id++, "Conv1", {3,3}, getFinalLayer(),13); //output 13*5*5
    addLayer(conv1);
    NormalizationLayer* norm1 = new NormalizationLayer(id++, "Norm1",getFinalLayer());
    addLayer(norm1);

    ConvolutionLayer* conv2 = new ConvolutionLayer(id++, "Conv2", {13,3,3}, getFinalLayer()); //output 3*3
    addLayer(conv2);

    VectorizationLayer* vec1 = new VectorizationLayer(id++, "Vec1", getFinalLayer());
    addLayer(vec1);

    FCLayer* fc1 = new FCLayer(id++, "FC1", 9, getFinalLayer());
    addLayer(fc1);

    LossConvexExample1* loss = new LossConvexExample1(id++, "Loss", getFinalLayer());
    addLayer(loss);
}
void ConvNet::train(){
    Tensor<float> inputTensor({7,7});
    InputLayer* inputLayer = getInputLayer();
    LossConvexExample1* lossLayer = (LossConvexExample1*) getFinalLayer();

    int batchSize = getBatchSize();
    int iBatch = 1000;

    long i=0;
    while (i< iBatch){
        zeroParaGradient();
        for(int j=0; j<batchSize; ++j){
            generateGaussian(&inputTensor,0,1);
            inputLayer->setInputTensor(inputTensor);
            forwardPropagate();
            backwardPropagate();
        }
        sgd(getLearningRate(), batchSize);
        printIteration(lossLayer, i);
        ++i;
    }
    lossLayer->printGroundTruth();
}
float ConvNet::test(){
   //null
}
