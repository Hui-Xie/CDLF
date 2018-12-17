//
// Created by Hui Xie on 9/18/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "CDLF.h"
#include "SegmentGNet.h"

SegmentGNet::SegmentGNet(const string& name, const string& saveDir): GNet(name, saveDir){

}

SegmentGNet::~SegmentGNet(){

}

// build method must assign the m_pInputXLayer, m_pGxLayer, m_pLossLayer
void SegmentGNet::build(){
    // Net uses load method to construct, this build  implement is obsolete.
    // it is a good design if all numFilter is odd;
    InputLayer* inputLayer01 = new InputLayer(1, "G_InputLayer1", {120,277, 277}); //output size: 120*277*277
    addLayer(inputLayer01);
    m_pInputXLayer = inputLayer01;

    NormalizationLayer* normAfterInput02 = new NormalizationLayer(5, "G_NormAfterInput5", inputLayer01);
    addLayer(normAfterInput02);

    ConvolutionLayer* conv30 = new ConvolutionLayer(30, "G_Conv30", normAfterInput02, {3,3, 3}, 3); //output size: 3*118*275*275
    addLayer(conv30);
    ReLU* reLU32 = new ReLU(32, "G_ReLU32", conv30);
    addLayer(reLU32);
    NormalizationLayer* norm34 = new NormalizationLayer(34, "G_Norm34", reLU32);
    addLayer(norm34);

    LinearLayer* bias40 = new LinearLayer(40, "G_Bias40", norm34); //output size: 3*118*275*275
    addLayer(bias40);

    BranchLayer* branch50 = new BranchLayer(50, "G_Branch50", bias40); //output size: 3*118*275*275
    addLayer(branch50);
    ReLU* reLU52 = new ReLU(52, "G_ReLU52", branch50);
    addLayer(reLU52);
    NormalizationLayer* norm54 = new NormalizationLayer(54, "G_Norm54", reLU52);
    addLayer(norm54);

    ConvolutionLayer* conv60 = new ConvolutionLayer(60, "G_Conv60", norm54, {3,3, 3,3}, 3); //output size: 3*116*273*273
    addLayer(conv60);
    ReLU* reLU62 = new ReLU(62, "G_ReLU62", conv60);
    addLayer(reLU62);
    NormalizationLayer* norm64 = new NormalizationLayer(64, "G_Norm64", reLU62);
    addLayer(norm64);

    ConvolutionLayer* conv70 = new ConvolutionLayer(70, "G_Conv70", norm64, {3,3,3,3}, 3); //output size: 3*114*271*271
    addLayer(conv70);
    ReLU* reLU4 = new ReLU(72, "G_ReLU72", conv70);
    addLayer(reLU4);
    NormalizationLayer* norm74 = new NormalizationLayer(74, "G_Norm74", reLU4);
    addLayer(norm74);

    ConvolutionLayer* conv80 = new ConvolutionLayer(80, "G_Conv80", norm74, {3,3,3,3}, 3); //output size: 3*112*269*269
    addLayer(conv80);
    ReLU* reLU82 = new ReLU(82, "G_ReLU82", conv80);
    addLayer(reLU82);
    NormalizationLayer* norm5 = new NormalizationLayer(84, "G_Norm85", reLU82);
    addLayer(norm5);

    ConvolutionLayer* conv90 = new ConvolutionLayer(90, "G_Conv90", norm5, {3,3,3,3},3); //output size: 3*110*267*267
    addLayer(conv90);
    ReLU* reLU92 = new ReLU(92, "G_ReLU92", conv90);
    addLayer(reLU92);
    NormalizationLayer* norm94 = new NormalizationLayer(94, "G_Norm94", reLU92);
    addLayer(norm94);


    ConvolutionLayer* conv100 = new ConvolutionLayer(100, "G_Conv100", norm94, {3,3,3,3}, 3); //output size: 3*108*265*265
    addLayer(conv100);
    ReLU* reLU102 = new ReLU(102, "G_ReLU102", conv100);
    addLayer(reLU102);
    NormalizationLayer* norm104 = new NormalizationLayer(104, "G_Norm104", reLU102);
    addLayer(norm104);

    //connect 2nd branch from branch1 to merger
    // branch1 output size: 7*118*275*275
    ConvolutionLayer* conv110 = new ConvolutionLayer(110, "G_Conv110", branch50, {3,3,3,3}, 3); //output size: 3*116*273*273
    addLayer(conv110);
    SubTensorLayer* subTensor111 = new SubTensorLayer(111, "G_SubTensor111", conv110, {0,4,4,4}, {3,108,265,265}); //output size: 3*108*265*265
    addLayer(subTensor111);
    ReLU* reLU112 = new ReLU(112, "G_ReLU112", subTensor111);
    addLayer(reLU112);
    NormalizationLayer* norm114 = new NormalizationLayer(114, "G_Norm114", reLU112);
    addLayer(norm114);

    MergerLayer* merger130 = new MergerLayer(130, "G_Merger130", {3,108,265,265}); //output size: 3*108*265*265
    addLayer(merger130);
    merger130->addPreviousLayer(norm104);
    merger130->addPreviousLayer(norm114);

    LinearLayer* bias140 = new LinearLayer(140, "G_Bias140", merger130); // output size: 3*108*265*265
    addLayer(bias140);

    SoftmaxLayer* softmax150 = new SoftmaxLayer(150, "G_Softmax150",bias140); //output size: 3*108*265*265
    addLayer(softmax150);

    BranchLayer* branch160 = new BranchLayer(160,"G_branch160", softmax150);//output size: 3*108*265*265
    addLayer(branch160);
    m_pGxLayer = branch160;

    CrossEntropyLoss* crossEntropy200 = new CrossEntropyLoss(200, "G_CrossEntropy200", branch160);
    addLayer(crossEntropy200);
    m_pLossLayer = crossEntropy200;

}

void SegmentGNet::train(){

}

float SegmentGNet::test(){
    return 0;
}