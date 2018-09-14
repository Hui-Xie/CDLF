//
// Created by Hui Xie on 9/14/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "Segmentation3DNet.h"



Segmentation3DNet::Segmentation3DNet(){

}

Segmentation3DNet::~Segmentation3DNet(){

}


void Segmentation3DNet::build(){
    buildG();
    buildD();
}


void Segmentation3DNet::buildG(){
    // it is a good design if all numFilter is odd;
    InputLayer* inputLayer = new InputLayer(1, "InputLayer", {277, 277,120}); //output size: 277*277*120
    addLayer(inputLayer);
    NormalizationLayer* normAfterInput = new NormalizationLayer(2, "NormAfterInput", inputLayer);
    addLayer(normAfterInput);

    BranchLayer* branch1 = new BranchLayer(20, "Branch1", normAfterInput);
    addLayer(branch1);

    ConvolutionLayer* conv1 = new ConvolutionLayer(30, "Conv1", branch1, {3,3, 3}, 5); //output size: 5*275*275*118
    addLayer(conv1);
    ReLU* reLU1 = new ReLU(32, "ReLU1", conv1);
    addLayer(reLU1);
    NormalizationLayer* norm1 = new NormalizationLayer(34, "Norm1", reLU1);
    addLayer(norm1);

    BiasLayer* bias1 = new BiasLayer(40, "Bias1", norm1); //output size: 5*275*275*118
    addLayer(bias1);

    BranchLayer* branch2 = new BranchLayer(50, "Branch2", bias1); //output size: 5*275*275*118
    addLayer(branch2);
    ReLU* reLU2 = new ReLU(52, "ReLU2", branch2);
    addLayer(reLU2);
    NormalizationLayer* norm2 = new NormalizationLayer(54, "Norm2", reLU2);
    addLayer(norm2);

    ConvolutionLayer* conv3 = new ConvolutionLayer(60, "Conv3", norm2, {5,5, 5,5}, 21); //output size: 21*271*271*114
    addLayer(conv3);
    ReLU* reLU3 = new ReLU(62, "ReLU3", conv3);
    addLayer(reLU3);
    NormalizationLayer* norm3 = new NormalizationLayer(64, "Norm3", reLU3);
    addLayer(norm3);

    ConvolutionLayer* conv4 = new ConvolutionLayer(70, "Conv4", norm3, {21,7,7,7}, 19); //output size: 19*265*265*108
    addLayer(conv4);
    ReLU* reLU4 = new ReLU(72, "ReLU4", conv4);
    addLayer(reLU4);
    NormalizationLayer* norm4 = new NormalizationLayer(74, "Norm4", reLU4);
    addLayer(norm4);


    ConvolutionLayer* conv5 = new ConvolutionLayer(80, "Conv5", norm4, {19,5,5,5}, 17); //output size: 17*261*261*104
    addLayer(conv5);
    ReLU* reLU5 = new ReLU(82, "ReLU5", conv5);
    addLayer(reLU5);
    NormalizationLayer* norm5 = new NormalizationLayer(84, "Norm5", reLU5);
    addLayer(norm5);


    ConvolutionLayer* conv6 = new ConvolutionLayer(90, "Conv6", norm5, {17,3,3,3},11); //output size: 11*259*259*102
    addLayer(conv6);
    ReLU* reLU6 = new ReLU(92, "ReLU6", conv6);
    addLayer(reLU6);
    NormalizationLayer* norm6 = new NormalizationLayer(94, "Norm6", reLU6);
    addLayer(norm6);


    ConvolutionLayer* conv7 = new ConvolutionLayer(100, "Conv7", norm6, {11,3,3,3}, 3); //output size: 3*257*257*100
    addLayer(conv7);
    ReLU* reLU7 = new ReLU(102, "ReLU7", conv7);
    addLayer(reLU7);
    NormalizationLayer* norm7 = new NormalizationLayer(104, "Norm7", reLU7);
    addLayer(norm7);


    SubTensorLayer* subTensor1 = new SubTensorLayer(110, "SubTensor1", branch2, {1,4,4,9},{3,257,257,100}); //output size: 3*257*257*100
    addLayer(subTensor1);
    ReLU* reLU8 = new ReLU(112, "ReLU8", subTensor1);
    addLayer(reLU8);
    NormalizationLayer* norm8 = new NormalizationLayer(114, "Norm8", reLU8);
    addLayer(norm8);

    MergerLayer* merger1 = new MergerLayer(120, "Merger1", {3,257,257,100}); //output size: 3*257*257*100
    addLayer(merger1);
    merger1->addPreviousLayer(norm7);
    merger1->addPreviousLayer(norm8);

    BiasLayer* bias2 = new BiasLayer(130, "Bias2", merger1); // output size: 3*257*257*100
    addLayer(bias2);

    SoftmaxLayer *softmax1 = new SoftmaxLayer(140, "Softmax1",bias2); //output size: 3*257*257*100
    addLayer(softmax1);

}

void Segmentation3DNet::buildD(){
    Layer* branch1 = getLayer(20); //  BranchLayer* branch1 = new BranchLayer(20, "Branch1", normAfterInput);

    SubTensorLayer* subTensor2 = new SubTensorLayer(540,"SubTensor2", branch1, {5,5,10}, {257,257,100});
    addLayer(subTensor2);

    ConvolutionLayer* conv21 = new ConvolutionLayer(550, "Conv21", subTensor2,{1,1,1},3);


}

void Segmentation3DNet::train(){
    InputLayer *inputLayer = getInputLayer();
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) getFinalLayer();

    long maxIteration =420;
    long NTrain = maxIteration;
    int batchSize = getBatchSize();
    float learningRate = getLearningRate();
    long numBatch = maxIteration / batchSize;
    if (0 != maxIteration % batchSize) {
        numBatch += 1;
    }

    long nIter = 0;
    long nBatch = 0;
    //random reshuffle data samples
    vector<long> randSeq = generateRandomSequence(NTrain);
    while (nBatch < numBatch) {
        zeroParaGradient();
        int i = 0;
        for (i = 0; i < batchSize && nIter < maxIteration; ++i) {
            //inputLayer->setInputTensor(m_pMnistData->m_pTrainImages->slice(randSeq[nIter]));
            //lossLayer->setGroundTruth(constructGroundTruth(m_pMnistData->m_pTrainLabels, randSeq[nIter]));
            forwardPropagate();
            backwardPropagate();
            ++nIter;
        }
        sgd(learningRate, i);
        ++nBatch;
    }
}

float Segmentation3DNet::test(){
    InputLayer *inputLayer = getInputLayer();
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) getFinalLayer();
    long n = 0;
    long nSuccess = 0;
    const long Ntest= 10000; //= m_pMnistData->m_pTestLabels->getLength();
    while (n < Ntest) {
        //inputLayer->setInputTensor(m_pMnistData->m_pTestImages->slice(n));
        //lossLayer->setGroundTruth(constructGroundTruth(m_pMnistData->m_pTestLabels, n));
        forwardPropagate();
        if (lossLayer->predictSuccessInColVec()) ++nSuccess;
        ++n;
    }
    cout<<"Info: nSuccess = "<<nSuccess<<" in "<<Ntest<<" test samples."<<endl;
    return  nSuccess * 1.0 / Ntest;
}

//construct a 2*1 one-hot vector
Tensor<float> Segmentation3DNet::constructGroundTruth(Tensor<unsigned char> *pLabels, const long index) {
    Tensor<float> tensor({10, 1});
    tensor.zeroInitialize();
    tensor.e(pLabels->e(index)) = 1;
    return tensor;
}
