//
// Created by Hui Xie on 9/18/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "CDLF.h"
#include "SegmentGNet.h"

SegmentGNet::SegmentGNet(const string& name): GNet(name){

}

SegmentGNet::~SegmentGNet(){

}

// build method must assign the m_pInputXLayer, m_pGxLayer, m_pLossLayer
void SegmentGNet::build(){
    // it is a good design if all numFilter is odd;
    InputLayer* inputLayer = new InputLayer(1, "G_InputLayer", {120,277, 277}); //output size: 120*277*277
    addLayer(inputLayer);
    m_pInputXLayer = inputLayer;

    NormalizationLayer* normAfterInput = new NormalizationLayer(2, "G_NormAfterInput", inputLayer);
    addLayer(normAfterInput);

    ConvolutionLayer* conv1 = new ConvolutionLayer(30, "G_Conv1", normAfterInput, {3,3, 3}, 7); //output size: 7*118*275*275
    addLayer(conv1);
    ReLU* reLU1 = new ReLU(32, "G_ReLU1", conv1);
    addLayer(reLU1);
    NormalizationLayer* norm1 = new NormalizationLayer(34, "G_Norm1", reLU1);
    addLayer(norm1);

    BiasLayer* bias1 = new BiasLayer(40, "G_Bias1", norm1); //output size: 7*118*275*275
    addLayer(bias1);

    BranchLayer* branch1 = new BranchLayer(50, "G_Branch1", bias1); //output size: 7*118*275*275
    addLayer(branch1);
    ReLU* reLU2 = new ReLU(52, "G_ReLU2", branch1);
    addLayer(reLU2);
    NormalizationLayer* norm2 = new NormalizationLayer(54, "G_Norm2", reLU2);
    addLayer(norm2);

    ConvolutionLayer* conv3 = new ConvolutionLayer(60, "G_Conv3", norm2, {7,3, 3,3}, 3); //output size: 3*116*273*273
    addLayer(conv3);
    ReLU* reLU3 = new ReLU(62, "G_ReLU3", conv3);
    addLayer(reLU3);
    NormalizationLayer* norm3 = new NormalizationLayer(64, "G_Norm3", reLU3);
    addLayer(norm3);

    ConvolutionLayer* conv4 = new ConvolutionLayer(70, "G_Conv4", norm3, {3,3,3,3}, 3); //output size: 3*114*271*271
    addLayer(conv4);
    ReLU* reLU4 = new ReLU(72, "G_ReLU4", conv4);
    addLayer(reLU4);
    NormalizationLayer* norm4 = new NormalizationLayer(74, "G_Norm4", reLU4);
    addLayer(norm4);

    ConvolutionLayer* conv5 = new ConvolutionLayer(80, "G_Conv5", norm4, {3,3,3,3}, 3); //output size: 3*112*269*269
    addLayer(conv5);
    ReLU* reLU5 = new ReLU(82, "G_ReLU5", conv5);
    addLayer(reLU5);
    NormalizationLayer* norm5 = new NormalizationLayer(84, "G_Norm5", reLU5);
    addLayer(norm5);

    ConvolutionLayer* conv6 = new ConvolutionLayer(90, "G_Conv6", norm5, {3,3,3,3},3); //output size: 3*110*267*267
    addLayer(conv6);
    ReLU* reLU6 = new ReLU(92, "G_ReLU6", conv6);
    addLayer(reLU6);
    NormalizationLayer* norm6 = new NormalizationLayer(94, "G_Norm6", reLU6);
    addLayer(norm6);


    ConvolutionLayer* conv7 = new ConvolutionLayer(100, "G_Conv7", norm6, {3,3,3,3}, 3); //output size: 3*108*265*265
    addLayer(conv7);
    ReLU* reLU7 = new ReLU(102, "G_ReLU7", conv7);
    addLayer(reLU7);
    NormalizationLayer* norm7 = new NormalizationLayer(104, "G_Norm7", reLU7);
    addLayer(norm7);

    //connect 2nd branch from branch1 to merger
    // branch1 output size: 7*118*275*275
    ConvolutionLayer* conv8 = new ConvolutionLayer(110, "G_Conv8", branch1, {7,3,3,3}, 3); //output size: 3*116*273*273
    addLayer(conv8);
    SubTensorLayer* subTensor8 = new SubTensorLayer(111, "G_SubTensor8", conv8, {0,4,4,4}, {3,108,265,265}); //output size: 3*108*265*265
    addLayer(subTensor8);
    ReLU* reLU8 = new ReLU(112, "G_ReLU8", subTensor8);
    addLayer(reLU8);
    NormalizationLayer* norm8 = new NormalizationLayer(114, "G_Norm8", reLU8);
    addLayer(norm8);

    MergerLayer* merger1 = new MergerLayer(130, "G_Merger1", {3,108,265,265}); //output size: 3*108*265*265
    addLayer(merger1);
    merger1->addPreviousLayer(norm7);
    merger1->addPreviousLayer(norm8);

    BiasLayer* bias2 = new BiasLayer(140, "G_Bias2", merger1); // output size: 3*108*265*265
    addLayer(bias2);

    SoftmaxLayer* softmax1 = new SoftmaxLayer(150, "G_Softmax1",bias2); //output size: 3*108*265*265
    addLayer(softmax1);

    BranchLayer* branch2 = new BranchLayer(160,"G_branch2", softmax1);//output size: 3*108*265*265
    addLayer(branch2);
    m_pGxLayer = branch2;

    CrossEntropyLoss* crossEntropy1 = new CrossEntropyLoss(200, "G_CrossEntropy1", branch2);
    addLayer(crossEntropy1);
    m_pLossLayer = crossEntropy1;

}

void SegmentGNet::train(){

}

float SegmentGNet::test(){
    return 0;
}