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
    InputLayer* inputLayer = new InputLayer(id++, "InputLayer", {17,17});  //output 17*17
    addLayer(inputLayer);

    ConvolutionLayer* conv1 = new ConvolutionLayer(id++, "Conv1", {3,3}, getFinalLayer(),7); //output 7*15*15
    addLayer(conv1);
    NormalizationLayer* norm1 = new NormalizationLayer(id++, "Norm1",getFinalLayer());
    addLayer(norm1);
    ReLU* reLU1 = new ReLU(id++, "ReLU1", getFinalLayer());
    addLayer(reLU1);

    ConvolutionLayer* conv2 = new ConvolutionLayer(id++, "Conv2", {3,3,3}, getFinalLayer(),5); //output 5*5*13*13
    addLayer(conv2);
    NormalizationLayer* norm2 = new NormalizationLayer(id++, "Norm2",getFinalLayer());
    addLayer(norm2);
    ReLU* reLU2 = new ReLU(id++, "ReLU2", getFinalLayer());
    addLayer(reLU2);

    ConvolutionLayer* conv3 = new ConvolutionLayer(id++, "Conv3", {3,3,3,3}, getFinalLayer(),1); //output 3*3*11*11
    addLayer(conv3);
    NormalizationLayer* norm3 = new NormalizationLayer(id++, "Norm3",getFinalLayer());
    addLayer(norm3);
    ReLU* reLU3 = new ReLU(id++, "ReLU3", getFinalLayer());
    addLayer(reLU3);

    ConvolutionLayer* conv4 = new ConvolutionLayer(id++, "Conv4", {3,3,5,5}, getFinalLayer(),1); //output 7*7
    addLayer(conv4);
    NormalizationLayer* norm4 = new NormalizationLayer(id++, "Norm4",getFinalLayer());
    addLayer(norm4);
    ReLU* reLU4 = new ReLU(id++, "ReLU4", getFinalLayer());
    addLayer(reLU4);

    ConvolutionLayer* conv5 = new ConvolutionLayer(id++, "Conv5", {3,3}, getFinalLayer(),1); //output 5*5
    addLayer(conv5);

    VectorizationLayer* vec1 = new VectorizationLayer(id++, "Vec1", getFinalLayer());
    addLayer(vec1);
    LossConvexExample1* loss = new LossConvexExample1(id++, "Loss", getFinalLayer());
    addLayer(loss);
}
void Conv4DNet::train(){
    Tensor<float> inputTensor({17,17});
    InputLayer* inputLayer = getInputLayer();
    LossConvexExample1* lossLayer = (LossConvexExample1*) getFinalLayer();

    long i=0;
    while (i< 500){
        zeroParaGradient();
        generateGaussian(&inputTensor,0,1);
        inputLayer->setInputTensor(inputTensor);
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
