//
// Created by Hui Xie on 8/17/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "ConvCudaNet.h"

ConvCudaNet::ConvCudaNet(const string& saveDir): FeedForwardNet(saveDir){

}
ConvCudaNet::~ConvCudaNet(){

}

void ConvCudaNet::build(){
    // build network
    int id =1;
    InputLayer* inputLayer = new InputLayer(id++, "InputLayer", {200,250,300});  //output 200*250*300
    addLayer(inputLayer);

    ConvolutionLayer* conv1 = new ConvolutionLayer(id++, "Conv1", getFinalLayer(), {3,3,3}, {1,1,1}); //output 198*248*298
    addLayer(conv1);
    NormalizationLayer* norm1 = new NormalizationLayer(id++, "Norm1",getFinalLayer(), getFinalLayer()->m_tensorSize);
    addLayer(norm1);
    ReLU* reLU1 = new ReLU(id++, "ReLU1", getFinalLayer(), getFinalLayer()->m_tensorSize);
    addLayer(reLU1);

    SoftmaxLayer* softmax = new SoftmaxLayer(id++, "Softmax", getFinalLayer());
    addLayer(softmax);

    CrossEntropyLoss* crossEntropy = new CrossEntropyLoss(id++, "CrossEntropy", getFinalLayer());
    addLayer(crossEntropy);

}
void ConvCudaNet::train(){
    InputLayer* inputLayer = getInputLayer();
    Tensor<float> inputTensor(inputLayer->m_pYTensor->getDims());
    CrossEntropyLoss* lossLayer = (CrossEntropyLoss*) getFinalLayer();

    int batchSize = getBatchSize();
    int iBatch = 30;

    Tensor<float> gt({198,248,298});
    generateGaussian(&gt,0,1);

    int i=0;
    while (i< iBatch){
        zeroParaGradient();
        for(int j=0; j<batchSize; ++j){
            generateGaussian(&inputTensor,0,1);
            inputLayer->setInputTensor(inputTensor);

            lossLayer->setGroundTruth(gt);
            forwardPropagate();

            //debug
            //printCurrentLocalTime();
            //cout<<"finished forward Propagate at "<< j<<" sample, in "<<i <<" batch"<<endl;
            //continue;


            backwardPropagate(true);
        }
        sgd(getLearningRate(), batchSize);
        cout<<"Info: ==============================finish batch: "<<i<<endl;
        //printIteration(lossLayer, i);
        ++i;
    }
    //lossLayer->printGroundTruth();
}
float ConvCudaNet::test(){
    //null
    return 0;
}
