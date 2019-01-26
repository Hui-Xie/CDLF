//
// Created by Hui Xie on 8/16/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "ConvNet.h"


ConvNet::ConvNet(const string& name, const string& saveDir): FeedForwardNet(name, saveDir){

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
    ReLU* reLU1 = new ReLU(id++, "ReLU1", getFinalLayer(), {13,6,5});
    addLayer(reLU1);
    NormalizationLayer* norm1 = new NormalizationLayer(id++, "Norm1",getFinalLayer(), {13,6,5});
    addLayer(norm1);


    ConvolutionLayer* conv2 = new ConvolutionLayer(id++, "Conv2",getFinalLayer(),  {13,3,3}, 10); //output 10*4*3
    addLayer(conv2);
    ReLU* reLU2 = new ReLU(id++, "ReLU2", getFinalLayer(), {10,4,3});
    addLayer(reLU2);
    NormalizationLayer* norm2 = new NormalizationLayer(id++, "Norm2",getFinalLayer(), {10,4,3});
    addLayer(norm2);

    ConvolutionLayer* conv3 = new ConvolutionLayer(id++, "Conv3", getFinalLayer(),{3,4,3}); //output 8*1
    addLayer(conv3);
    ReLU* reLU3 = new ReLU(id++, "ReLU3", getFinalLayer(), {8,1});
    addLayer(reLU3);
    NormalizationLayer* norm3 = new NormalizationLayer(id++, "Norm3",getFinalLayer(), {8,1});
    addLayer(norm3);

    VectorizationLayer* vec1 = new VectorizationLayer(id++, "Vec1", getFinalLayer()); // output 8*1
    addLayer(vec1);
    FCLayer* fc1 = new FCLayer(id++, "FC1", getFinalLayer(), 7);
    addLayer(fc1);
    MeanSquareLossLayer * loss = new MeanSquareLossLayer(id++, "Loss", getFinalLayer(), 1);
    addLayer(loss);
}
void ConvNet::train(){
    InputLayer* inputLayer = getInputLayer();
    Tensor<float> inputTensor(inputLayer->m_pYTensor->getDims());
    MeanSquareLossLayer* lossLayer = (MeanSquareLossLayer*) getFinalLayer();
    Tensor<float> groundTruth(lossLayer->m_prevLayer->m_tensorSize);
    for (int i=0; i< groundTruth.getLength(); ++i){
        groundTruth.e(i) = i;
    }
    lossLayer->setGroundTruth(groundTruth);

    int batchSize = getBatchSize();
    int iBatch = 130;

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
float ConvNet::test(){
   //null
   return 0;
}
