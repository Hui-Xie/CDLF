//
// Created by Hui Xie on 9/21/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#include "StubNetForD.h"
#include "SoftmaxLayer.h"
#include "InputLayer.h"
#include "statisTool.h"


StubNetForD::StubNetForD(const string& saveDir): FeedForwardNet(saveDir){

}

StubNetForD::~StubNetForD(){

}

void StubNetForD::build(){
    // Net uses load method to construct, this build  implement is obsolete.

    InputLayer* inputLayer01 = new InputLayer(1, "S_InputLayer01", {3,108,265,265});
    addLayer(inputLayer01);

    SoftmaxLayer* softmax10 = new SoftmaxLayer(10, "S_Softmax10",inputLayer01); //output size: 3*108*265*265
    addLayer(softmax10);
}

void StubNetForD::train(){
   //null
}

float StubNetForD::test(){
   //null
}

void  StubNetForD::randomOutput(){
    generateGaussian(getInputLayer()->m_pYTensor, 0,1);
    forwardPropagate();
}

Tensor<float>* StubNetForD::getOutput(){
    return getFinalLayer()->m_pYTensor;
}
