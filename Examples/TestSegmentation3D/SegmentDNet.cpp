//
// Created by Hui Xie on 9/18/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "SegmentDNet.h"
#include "CDLF.h"

SegmentDNet::SegmentDNet(const string &name, const string& saveDir) : DNet(name, saveDir) {

}

SegmentDNet::~SegmentDNet() {

}

// build method must assign m_pGTLayer, m_pGxLayer, m_pInputXLayer, m_pMerger, m_pLossLayer;
void SegmentDNet::build() {
    // Net uses load method to construct, this build  implement is obsolete.

    m_pInputXLayer = new InputLayer(1, "D_InputLayer1", {120, 277, 277});
    addLayer(m_pInputXLayer);
    NormalizationLayer *normAfterInput04 = new NormalizationLayer(6, "D_NormAfterInput6", m_pInputXLayer);
    addLayer(normAfterInput04);
    SubTensorLayer *subTensor10 = new SubTensorLayer(10, "D_SubTensor10", normAfterInput04, {6, 6, 6},
                                                     {108, 265, 265}); //output size: 108*265*265
    addLayer(subTensor10);
    //let this convolution layer to learn best parameter to match the probability
    ConvolutionLayer *conv20 = new ConvolutionLayer(20, "D_Conv20", subTensor10, {1, 1, 1}, 3); //output: 3*108*265*265
    addLayer(conv20);
    NormalizationLayer *norm24 = new NormalizationLayer(24, "D_Norm24", conv20);
    addLayer(norm24);
    ScaleLayer *scale26 = new ScaleLayer(26, "D_Scale26", norm24);
    addLayer(scale26);


    m_pGTLayer = new InputLayer(2, "D_GroundTruthLayer2", {3, 108, 265, 265}); //output size: 3*108*265*265
    addLayer(m_pGTLayer);

    m_pGxLayer = new InputLayer(3, "D_GxLayer3", {3, 108, 265, 265}); //output size: 3*108*265*265
    addLayer(m_pGxLayer);

    m_pMerger = new MergerLayer(30, "D_Merger30", {3, 108, 265, 265});//output size: 3*108*265*265
    addLayer(m_pMerger);
    m_pMerger->addPreviousLayer(scale26);
    m_pMerger->addPreviousLayer(m_pGTLayer);
    m_pMerger->addPreviousLayer(m_pGxLayer);

    ConvolutionLayer *conv40 = new ConvolutionLayer(40, "D_Conv40", m_pMerger, {3, 3, 3, 3}, 3);//outputsize: 3*106*263*263
    addLayer(conv40);
    ReLU *reLU42 = new ReLU(42, "D_ReLU42", conv40);
    addLayer(reLU42);
    NormalizationLayer *norm44 = new NormalizationLayer(44, "D_norm44", reLU42);
    addLayer(norm44);


    ConvolutionLayer *conv50 = new ConvolutionLayer(50, "D_Conv50", norm44, {3, 3, 3, 3}, 3);//outputsize: 3*104*261*261
    addLayer(conv50);
    ReLU *reLU52 = new ReLU(52, "D_ReLU52", conv50);
    addLayer(reLU52);
    NormalizationLayer *norm54 = new NormalizationLayer(54, "D_norm54", reLU52);
    addLayer(norm54);

    ConvolutionLayer *conv60 = new ConvolutionLayer(60, "D_Conv60", norm54, {3, 3, 3, 3}, 1);//outputsize: 102*259*259
    addLayer(conv60);
    ReLU *reLU62 = new ReLU(62, "D_ReLU62", conv60);
    addLayer(reLU62);
    NormalizationLayer *norm64 = new NormalizationLayer(64, "D_norm64", reLU62);
    addLayer(norm64);

    ConvolutionLayer *conv70 = new ConvolutionLayer(70, "D_Conv70", norm64, {3, 3, 3}, 1);//outputsize: 100*257*257
    addLayer(conv70);
    ReLU *reLU72 = new ReLU(72, "D_ReLU72", conv70);
    addLayer(reLU72);
    NormalizationLayer *norm74 = new NormalizationLayer(74, "D_norm74", reLU72);
    addLayer(norm74);

    MaxPoolingLayer *maxPool76 = new MaxPoolingLayer(76, "D_maxPool76", norm74, {3, 3, 3}, 3); //outputsize: 33*85*85
    addLayer(maxPool76);

    ConvolutionLayer *conv80 = new ConvolutionLayer(80, "D_Conv80", maxPool76, {3, 3, 3}, 1);//outputsize: 32*83*83
    addLayer(conv80);
    ReLU *reLU82 = new ReLU(82, "D_ReLU82", conv80);
    addLayer(reLU82);
    NormalizationLayer *norm84 = new NormalizationLayer(84, "D_norm84", reLU82);
    addLayer(norm84);
    MaxPoolingLayer *maxPool86 = new MaxPoolingLayer(86, "D_maxPool86", norm84, {3, 3, 3}, 3); //outputsize: 10*27*27
    addLayer(maxPool86);

    ConvolutionLayer *conv90 = new ConvolutionLayer(90, "D_Conv90", maxPool86, {10, 3, 3}, 1);//outputsize: 25*25
    addLayer(conv90);
    ReLU *reLU92 = new ReLU(92, "D_ReLU92", conv90);
    addLayer(reLU92);
    NormalizationLayer *norm94 = new NormalizationLayer(94, "D_norm94", reLU92);
    addLayer(norm94);

    VectorizationLayer *vec110 = new VectorizationLayer(110, "D_Vectorization110", norm94); // outputsize: 625*1
    addLayer(vec110);

    FCLayer *fc120 = new FCLayer(120, "D_fc120", vec110, 400); //outputsize 400*1
    addLayer(fc120);
    ReLU *reLU122 = new ReLU(122, "D_ReLU122", fc120);
    addLayer(reLU122);
    NormalizationLayer *norm124 = new NormalizationLayer(124, "D_norm124", reLU122);
    addLayer(norm124);

    FCLayer *fc130 = new FCLayer(130, "D_fc130", norm124, 200); //outputsize 200*1
    addLayer(fc130);
    ReLU *reLU132 = new ReLU(132, "D_ReLU132", fc130);
    addLayer(reLU132);
    NormalizationLayer *norm134 = new NormalizationLayer(134, "D_norm134", reLU132);
    addLayer(norm134);


    FCLayer *fc140 = new FCLayer(140, "D_fc140", norm134, 50); //outputsize 50*1
    addLayer(fc140);
    ReLU *reLU142 = new ReLU(142, "D_ReLU142", fc140);
    addLayer(reLU142);
    NormalizationLayer *norm144 = new NormalizationLayer(144, "D_norm144", reLU142);
    addLayer(norm144);

    FCLayer *fc150 = new FCLayer(150, "D_fc150", norm144, 10); //outputsize 10*1
    addLayer(fc150);
    ReLU *reLU152 = new ReLU(152, "D_ReLU152", fc150);
    addLayer(reLU152);
    NormalizationLayer *norm154 = new NormalizationLayer(154, "D_norm154", reLU152);
    addLayer(norm154);


    FCLayer *fc160 = new FCLayer(160, "D_fc160", norm154, 2); //outputsize 2*1
    addLayer(fc160);
    ReLU *reLU162 = new ReLU(162, "D_ReLU162", fc160);
    addLayer(reLU162);
    NormalizationLayer *norm164 = new NormalizationLayer(164, "D_norm164", reLU162);
    addLayer(norm164);

    SoftmaxLayer *softmax170 = new SoftmaxLayer(170, "D_softmax170", norm164);
    addLayer(softmax170);

    CrossEntropyLoss *crossEntropy200 = new CrossEntropyLoss(200, "D_CrossEntropy200", softmax170);//outputsize 2*1
    addLayer(crossEntropy200);
    m_pLossLayer = crossEntropy200;

}


void SegmentDNet::train() {

}

float SegmentDNet::test() {
    return 0;
}
