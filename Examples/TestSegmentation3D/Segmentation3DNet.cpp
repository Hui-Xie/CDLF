//
// Created by Hui Xie on 9/14/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "Segmentation3DNet.h"



Segmentation3DNet::Segmentation3DNet(){

}

Segmentation3DNet::~Segmentation3DNet(){

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

    SoftmaxLayer* softmax1 = new SoftmaxLayer(140, "Softmax1",bias2); //output size: 3*257*257*100
    addLayer(softmax1);

    BranchLayer* branch3 = new BranchLayer(150,"branch3", softmax1);
    addLayer(branch3);

    CrossEntropyLoss* crossEntropy1 = new CrossEntropyLoss(160, "CrossEntropy1", branch3);
    addLayer(crossEntropy1);

}

void Segmentation3DNet::buildD(){

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
    //todo: merger2 needs to addPreviousLayer of G's result or Groundtruth


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

void Segmentation3DNet::trainG(){
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
            //forwardPropagate();
            //backwardPropagate();
            ++nIter;
        }
        //sgd(learningRate, i);
        ++nBatch;
    }
}

void Segmentation3DNet::trainD(){


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
        //forwardPropagate();
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
