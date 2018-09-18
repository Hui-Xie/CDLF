//
// Created by Hui Xie on 9/18/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "DNet.h"
#include "CDLF.h"

DNet::DNet(const string& name): FeedForwardNet(name) {
    m_pGTLayer = nullptr;
    m_pGxLayer = nullptr;
    m_pInputXLayer = nullptr;
}

DNet::~DNet() {

}

void DNet::build(){

    m_pGTLayer = new InputLayer(151, "GroundTruthLayer", {277, 277,120}); //output size: 277*277*120
    addLayer(m_pGTLayer);

    //extract original input for D
    Layer* branch1 = getLayer(20); //  BranchLayer* branch1 = new BranchLayer(20, "Branch1", normAfterInput);

    SubTensorLayer* subTensor2 = new SubTensorLayer(540,"SubTensor2", branch1, {5,5,10}, {257,257,100});
    addLayer(subTensor2);

    //todo: this conv21 needs to fixed parameter. Namely non-learning parameter
    ConvolutionLayer* conv21 = new ConvolutionLayer(550, "Conv21", subTensor2,{1,1,1},3); //output: 3*257*257*100
    addLayer(conv21);
    NormalizationLayer* norm21 = new NormalizationLayer(554, "Norm21", conv21);
    addLayer(norm21);

    MergerLayer* merger2 = new MergerLayer(600, "Merger2", {3,257,257,100});
    addLayer(merger2);
    merger2->addPreviousLayer(norm21);
    //merger2->addPreviousLayer(m_pGroundTruthLayer);

    ConvolutionLayer* conv22 = new ConvolutionLayer(610,"Conv22", merger2,{3,31,31,31},15);//outputsize: 15*227*227*70
    addLayer(conv22);
    ReLU* reLU22 = new ReLU(612, "ReLU22", conv22);
    addLayer(reLU22);
    NormalizationLayer* norm22 = new NormalizationLayer(614,"norm22", reLU22);
    addLayer(norm22);

    ConvolutionLayer* conv23 = new ConvolutionLayer(620,"Conv23", norm22,{15,27,27,27},11);//outputsize: 11*201*201*44
    addLayer(conv23);
    ReLU* reLU23 = new ReLU(622, "ReLU23", conv23);
    addLayer(reLU23);
    NormalizationLayer* norm23 = new NormalizationLayer(624,"norm23", reLU23);
    addLayer(norm23);

    ConvolutionLayer* conv24 = new ConvolutionLayer(630,"Conv24", norm23,{11,21,21,21},7);//outputsize: 7*181*181*24
    addLayer(conv24);
    ReLU* reLU24 = new ReLU(632, "ReLU24", conv24);
    addLayer(reLU24);
    NormalizationLayer* norm24 = new NormalizationLayer(634,"norm24", reLU24);
    addLayer(norm24);

    ConvolutionLayer* conv25 = new ConvolutionLayer(640,"Conv25", norm24,{7,33,33,24},1);//outputsize: 149*149
    addLayer(conv25);
    ReLU* reLU25 = new ReLU(642, "ReLU25", conv25);
    addLayer(reLU25);
    NormalizationLayer* norm25 = new NormalizationLayer(644,"norm25", reLU25);
    addLayer(norm25);

    ConvolutionLayer* conv26 = new ConvolutionLayer(650,"Conv26", norm25,{41,41},1);//outputsize: 109*109
    addLayer(conv26);
    ReLU* reLU26 = new ReLU(652, "ReLU26", conv26);
    addLayer(reLU26);
    NormalizationLayer* norm26 = new NormalizationLayer(654,"norm26", reLU26);
    addLayer(norm26);

    ConvolutionLayer* conv27 = new ConvolutionLayer(660,"Conv27", norm26,{25,25});//outputsize: 85*85
    addLayer(conv27);
    ReLU* reLU27 = new ReLU(662, "ReLU27", conv27);
    addLayer(reLU27);
    NormalizationLayer* norm27 = new NormalizationLayer(664,"norm27", reLU27);
    addLayer(norm27);

    ConvolutionLayer* conv28 = new ConvolutionLayer(670,"Conv28", norm27,{23,23});//outputsize: 63*63
    addLayer(conv28);
    ReLU* reLU28 = new ReLU(672, "ReLU28", conv28);
    addLayer(reLU28);
    NormalizationLayer* norm28 = new NormalizationLayer(674,"norm28", reLU28);
    addLayer(norm28);

    VectorizationLayer* vec1= new VectorizationLayer(680, "Vectorization1", norm28); // outputsize: 3969*1
    addLayer(vec1);

    FCLayer* fc41 = new FCLayer(690, "fc41", vec1, 500); //outputsize 500*1
    addLayer(fc41);
    ReLU* reLU41 = new ReLU(692, "ReLU41", fc41);
    addLayer(reLU41);
    NormalizationLayer* norm41 = new NormalizationLayer(694,"norm41", reLU41);
    addLayer(norm41);

    FCLayer* fc42 = new FCLayer(700, "fc42", norm41, 200); //outputsize 200*1
    addLayer(fc42);
    ReLU* reLU42 = new ReLU(702, "ReLU42", fc42);
    addLayer(reLU42);
    NormalizationLayer* norm42 = new NormalizationLayer(704,"norm42", reLU42);
    addLayer(norm42);


    FCLayer* fc43 = new FCLayer(710, "fc43", norm42, 50); //outputsize 50*1
    addLayer(fc43);
    ReLU* reLU43 = new ReLU(712, "ReLU43", fc43);
    addLayer(reLU43);
    NormalizationLayer* norm43 = new NormalizationLayer(714,"norm43", reLU43);
    addLayer(norm43);

    FCLayer* fc44 = new FCLayer(720, "fc44", norm43, 10); //outputsize 10*1
    addLayer(fc44);
    ReLU* reLU44 = new ReLU(722, "ReLU44", fc44);
    addLayer(reLU44);
    NormalizationLayer* norm44 = new NormalizationLayer(724,"norm44", reLU44);
    addLayer(norm44);


    FCLayer* fc45 = new FCLayer(730, "fc45", norm44, 2); //outputsize 2*1
    addLayer(fc45);
    ReLU* reLU45 = new ReLU(732, "ReLU45", fc45);
    addLayer(reLU45);
    NormalizationLayer* norm45 = new NormalizationLayer(734,"norm45", reLU45);
    addLayer(norm45);

    SoftmaxLayer* softmax2 = new SoftmaxLayer(740, "softmax2", norm45);
    addLayer(softmax2);

    CrossEntropyLoss* crossEntropy2 = new CrossEntropyLoss(750, "CrossEntropy2", softmax2);
    addLayer(crossEntropy2);

}


void DNet::train(){

}

float DNet::test(){
    return 0;
}
