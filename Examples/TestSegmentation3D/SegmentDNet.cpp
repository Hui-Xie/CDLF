//
// Created by Hui Xie on 9/18/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "SegmentDNet.h"
#include "CDLF.h"

SegmentDNet::SegmentDNet(const string& name): DNet(name) {

}

SegmentDNet::~SegmentDNet() {

}

// build method must assign m_pGTLayer, m_pGxLayer, m_pInputXLayer, m_pMerger, m_pLossLayer;
void SegmentDNet::build(){
    m_pInputXLayer = new InputLayer(1,"OriginalInputLayer", {120,277,277});
    addLayer(m_pInputXLayer);
    NormalizationLayer* normAfterInput = new NormalizationLayer(4, "NormAfterInput", m_pInputXLayer);
    addLayer(normAfterInput);
    SubTensorLayer* subTensor1 = new SubTensorLayer(10,"SubTensor1", normAfterInput, {6,6,6}, {108,265,265}); //output size: 108*265*265
    addLayer(subTensor1);
    //let this convolution layer to learn best parameter to match the probability
    ConvolutionLayer* conv20 = new ConvolutionLayer(20, "Conv20", subTensor1,{1,1,1},3); //output: 3*108*265*265
    addLayer(conv20);
    NormalizationLayer* norm24 = new NormalizationLayer(24, "Norm24", conv20);
    addLayer(norm24);
    ScaleLayer* scale1 =new ScaleLayer(26, "Scale26", norm24);
    addLayer(scale1);


    m_pGTLayer = new InputLayer(0, "GroundTruthLayer", {3, 108,265, 265}); //output size: 3*108*265*265
    addLayer(m_pGTLayer);

    m_pGxLayer = new InputLayer(2, "GxLayer", {3, 108,265, 265}); //output size: 3*108*265*265
    addLayer(m_pGxLayer);

    m_pMerger = new MergerLayer(30, "Merger1", {3,108,265,265});//output size: 3*108*265*265
    addLayer(m_pMerger);
    m_pMerger->addPreviousLayer(scale1);
    m_pMerger->addPreviousLayer(m_pGTLayer);
    m_pMerger->addPreviousLayer(m_pGxLayer);

    ConvolutionLayer* conv40 = new ConvolutionLayer(40,"Conv40", m_pMerger,{3,3,3,3},21);//outputsize: 21*106*263*263
    addLayer(conv40);
    ReLU* reLU42 = new ReLU(42, "ReLU42", conv40);
    addLayer(reLU42);
    NormalizationLayer* norm44 = new NormalizationLayer(44,"norm44", reLU42);
    addLayer(norm44);

    ConvolutionLayer* conv50 = new ConvolutionLayer(50,"Conv50", norm44,{21,3,3,3},21);//outputsize: 21*104*261*261
    addLayer(conv50);
    ReLU* reLU52 = new ReLU(52, "ReLU52", conv50);
    addLayer(reLU52);
    NormalizationLayer* norm54 = new NormalizationLayer(54,"norm54", reLU52);
    addLayer(norm54);

    ConvolutionLayer* conv60 = new ConvolutionLayer(60,"Conv60", norm54,{21,3,3,3},1);//outputsize: 102*259*259
    addLayer(conv60);
    ReLU* reLU62 = new ReLU(62, "ReLU62", conv60);
    addLayer(reLU62);
    NormalizationLayer* norm64 = new NormalizationLayer(64,"norm64", reLU62);
    addLayer(norm64);

    ConvolutionLayer* conv70 = new ConvolutionLayer(70,"Conv70", norm64,{3,3,3},1);//outputsize: 100*257*257
    addLayer(conv70);
    ReLU* reLU72 = new ReLU(72, "ReLU72", conv70);
    addLayer(reLU72);
    NormalizationLayer* norm74 = new NormalizationLayer(74,"norm74", reLU72);
    addLayer(norm74);
    MaxPoolingLayer* maxPool76 = new MaxPoolingLayer(76, "maxPool76", norm74, {3,3,3}, 3); //outputsize: 33*85*85
    addLayer(maxPool76);

    ConvolutionLayer* conv80 = new ConvolutionLayer(80,"Conv80", maxPool76, {3,3,3},1);//outputsize: 32*83*83
    addLayer(conv80);
    ReLU* reLU82 = new ReLU(82, "ReLU82", conv80);
    addLayer(reLU82);
    NormalizationLayer* norm84 = new NormalizationLayer(84,"norm84", reLU82);
    addLayer(norm84);
    MaxPoolingLayer* maxPool86 = new MaxPoolingLayer(86, "maxPool86", norm84, {3,3,3}, 3); //outputsize: 10*27*27
    addLayer(maxPool86);

    ConvolutionLayer* conv90 = new ConvolutionLayer(90,"Conv90", maxPool86,{10,3,3}, 1);//outputsize: 25*25
    addLayer(conv90);
    ReLU* reLU92 = new ReLU(92, "ReLU92", conv90);
    addLayer(reLU92);
    NormalizationLayer* norm94 = new NormalizationLayer(94,"norm94", reLU92);
    addLayer(norm94);

    VectorizationLayer* vec1= new VectorizationLayer(110, "Vectorization1", norm94); // outputsize: 625*1
    addLayer(vec1);

    FCLayer* fc120 = new FCLayer(120, "fc120", vec1, 400); //outputsize 400*1
    addLayer(fc120);
    ReLU* reLU122 = new ReLU(122, "ReLU122", fc120);
    addLayer(reLU122);
    NormalizationLayer* norm124 = new NormalizationLayer(124,"norm124", reLU122);
    addLayer(norm124);

    FCLayer* fc130 = new FCLayer(130, "fc130", norm124, 200); //outputsize 200*1
    addLayer(fc130);
    ReLU* reLU132 = new ReLU(132, "ReLU132", fc130);
    addLayer(reLU132);
    NormalizationLayer* norm134 = new NormalizationLayer(134,"norm134", reLU132);
    addLayer(norm134);


    FCLayer* fc140 = new FCLayer(140, "fc140", norm134, 50); //outputsize 50*1
    addLayer(fc140);
    ReLU* reLU142 = new ReLU(142, "ReLU142", fc140);
    addLayer(reLU142);
    NormalizationLayer* norm144 = new NormalizationLayer(144,"norm144", reLU142);
    addLayer(norm144);

    FCLayer* fc150 = new FCLayer(150, "fc150", norm144, 10); //outputsize 10*1
    addLayer(fc150);
    ReLU* reLU152 = new ReLU(152, "ReLU152", fc150);
    addLayer(reLU152);
    NormalizationLayer* norm154 = new NormalizationLayer(154,"norm154", reLU152);
    addLayer(norm154);


    FCLayer* fc160 = new FCLayer(160, "fc160", norm154, 2); //outputsize 2*1
    addLayer(fc160);
    ReLU* reLU162 = new ReLU(162, "ReLU162", fc160);
    addLayer(reLU162);
    NormalizationLayer* norm164 = new NormalizationLayer(164,"norm164", reLU162);
    addLayer(norm164);

    SoftmaxLayer* softmax1 = new SoftmaxLayer(170, "softmax1", norm164);
    addLayer(softmax1);

    CrossEntropyLoss* crossEntropy1 = new CrossEntropyLoss(180, "CrossEntropy1", softmax1);//outputsize 2*1
    addLayer(crossEntropy1);
    m_pLossLayer = crossEntropy1;

}


void SegmentDNet::train(){

}

float SegmentDNet::test(){
    return 0;
}
