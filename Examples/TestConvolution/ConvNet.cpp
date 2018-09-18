//
// Created by Hui Xie on 8/16/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "ConvNet.h"
#include "LossConvexExample1.h"

ConvNet::ConvNet(const string& name): FeedForwardNet(name){

}
ConvNet::~ConvNet(){

}

void ConvNet::build(){
    // build network
    int id =1;
    InputLayer* inputLayer = new InputLayer(id++, "InputLayer", {8,7});  //output: 8*7
    addLayer(inputLayer);

    ConvolutionLayer* conv1 = new ConvolutionLayer(id++, "Conv1", getFinalLayer(),{3,3}, 13); //output 13*6*5
    addLayer(conv1);
    NormalizationLayer* norm1 = new NormalizationLayer(id++, "Norm1",getFinalLayer());
    addLayer(norm1);
    ReLU* reLU1 = new ReLU(id++, "ReLU1", getFinalLayer());
    addLayer(reLU1);

    ConvolutionLayer* conv2 = new ConvolutionLayer(id++, "Conv2",getFinalLayer(),  {13,3,3}, 10); //output 10*4*3
    addLayer(conv2);
    NormalizationLayer* norm2 = new NormalizationLayer(id++, "Norm2",getFinalLayer());
    addLayer(norm2);
    ReLU* reLU2 = new ReLU(id++, "ReLU2", getFinalLayer());
    addLayer(reLU2);


    ConvolutionLayer* conv3 = new ConvolutionLayer(id++, "Conv3", getFinalLayer(),{3,4,3}); //output 8*1
    addLayer(conv3);
    NormalizationLayer* norm3 = new NormalizationLayer(id++, "Norm3",getFinalLayer());
    addLayer(norm3);
    ReLU* reLU3 = new ReLU(id++, "ReLU3", getFinalLayer());
    addLayer(reLU3);


    VectorizationLayer* vec1 = new VectorizationLayer(id++, "Vec1", getFinalLayer()); // output 8*1
    addLayer(vec1);
    FCLayer* fc1 = new FCLayer(id++, "FC1", getFinalLayer(), 7);
    addLayer(fc1);
    LossConvexExample1* loss = new LossConvexExample1(id++, "Loss", getFinalLayer());
    addLayer(loss);
}
void ConvNet::train(){
    InputLayer* inputLayer = getInputLayer();
    Tensor<float> inputTensor(inputLayer->m_pYTensor->getDims());
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
   return 0;
}
