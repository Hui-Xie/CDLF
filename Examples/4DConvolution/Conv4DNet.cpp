//
// Created by Hui Xie on 8/17/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "Conv4DNet.h"
#include "LossConvexExample1.h"

Conv4DNet::Conv4DNet(){

}
Conv4DNet::~Conv4DNet(){

}

void Conv4DNet::build(){
    // build network
    int id =1;
    InputLayer* inputLayer = new InputLayer(id++, "InputLayer", {7,7});
    addLayer(inputLayer);

    ConvolutionLayer* conv1 = new ConvolutionLayer(id++, "Conv1", {3,3}, getFinalLayer(),3); //output 3*5*5
    addLayer(conv1);

    NormalizationLayer* norm1 = new NormalizationLayer(id++, "Norm1",getFinalLayer());
    addLayer(norm1);

    ConvolutionLayer* conv2 = new ConvolutionLayer(id++, "Conv2", {3,3,3}, getFinalLayer()); //output 3*3
    addLayer(conv2);

    VectorizationLayer* vec1 = new VectorizationLayer(id++, "Vec1", getFinalLayer());
    addLayer(vec1);

    LossConvexExample1* loss = new LossConvexExample1(id++, "Loss", getFinalLayer());
    addLayer(loss);
}
void Conv4DNet::train(){
    Tensor<float> inputTensor({7,7});
    generateGaussian(&inputTensor,0,1);
    InputLayer* inputLayer = getInputLayer();
    inputLayer->setInputTensor(inputTensor);
    LossConvexExample1* lossLayer = (LossConvexExample1*) getFinalLayer();

    long i=0;
    while (i< 200){
        zeroParaGradient();
        forwardPropagate();
        printIteration(lossLayer, i);
        backwardPropagate();
        sgd(getLearningRate(), 1);
        ++i;
    }
    lossLayer->printGroundTruth();
}
float Conv4DNet::test(){
    //null
}
