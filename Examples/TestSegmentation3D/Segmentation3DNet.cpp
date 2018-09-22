//
// Created by Hui Xie on 9/14/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "Segmentation3DNet.h"



Segmentation3DNet::Segmentation3DNet(const string& name, GNet* pGNet, DNet* pDNet): GAN(name, pGNet, pDNet){

}

Segmentation3DNet::~Segmentation3DNet(){

}

void Segmentation3DNet::quicklySwitchTrainG_D(){
    const long N = m_pDataMgr->m_NTrainFile;
    const int batchSize = m_pGNet->getBatchSize();
    long numBatch = N / batchSize;
    if (0 != N % batchSize) {
        numBatch += 1;
    }

    long nIter = 0;
    long batch = 0;
    vector<long> randSeq = generateRandomSequence(N);
    while (batch < numBatch) {
        m_pGNet->zeroParaGradient();
        m_pDNet->zeroParaGradient();
        int i = 0;
        int ignoreGx = 0;
        for (i = 0; i < batchSize && nIter < N; ++i) {
            long index = randSeq[nIter];

            Tensor<float> *pImage= nullptr;
            m_pDataMgr->readTrainImageFile(index, pImage);
            m_pGNet->m_pInputXLayer->setInputTensor(*pImage);
            m_pDNet->m_pInputXLayer->setInputTensor(*pImage);
            delete pImage;

            Tensor<unsigned char> *pLabel= nullptr;
            m_pDataMgr->readTrainLabelFile(index, pLabel);
            Tensor<float>* pOneHotLabel= nullptr;
            m_pDataMgr->oneHotEncodeLabel(pLabel, pOneHotLabel, 3);
            delete pLabel;
            m_pGNet->m_pLossLayer->setGroundTruth(*pOneHotLabel);
            m_pDNet->m_pGTLayer->setInputTensor(*pOneHotLabel);
            delete pOneHotLabel;

            // train G
            m_pDNet->setAlphaGroundTruth(true);
            switchDtoGx();
            forwardG();
            backwardG();

            // train D
            switchDtoGT();
            forwardD();
            backwardD();

            if (*m_pGNet->m_pGxLayer->m_pYTensor != *m_pDNet->m_pGTLayer->m_pYTensor){
                m_pDNet->setAlphaGroundTruth(false);
                switchDtoGx();
                forwardD();
                backwardD();
            }
            else{
                ++ignoreGx;
            }
            ++nIter;
        }
        m_pGNet->sgd(m_pGNet->getLearningRate(), i);
        m_pDNet->sgd(m_pDNet->getLearningRate(), i*2-ignoreGx);
        ++batch;
    }
}

void Segmentation3DNet::trainG(){
    const long N = m_pDataMgr->m_NTrainFile;
    const int batchSize = m_pGNet->getBatchSize();
    long numBatch = N / batchSize;
    if (0 != N % batchSize) {
        numBatch += 1;
    }

    long nIter = 0;
    long batch = 0;
    vector<long> randSeq = generateRandomSequence(N);

    m_pDNet->setAlphaGroundTruth(true);
    switchDtoGx();

    while (batch < numBatch) {
        m_pGNet->zeroParaGradient();
        int i = 0;
        for (i = 0; i < batchSize && nIter < N; ++i) {
            long index = randSeq[nIter];

            Tensor<float> *pImage= nullptr;
            m_pDataMgr->readTrainImageFile(index, pImage);
            m_pGNet->m_pInputXLayer->setInputTensor(*pImage);
            m_pDNet->m_pInputXLayer->setInputTensor(*pImage);
            delete pImage;

            Tensor<unsigned char> *pLabel= nullptr;
            m_pDataMgr->readTrainLabelFile(index, pLabel);
            Tensor<float>* pOneHotLabel= nullptr;
            m_pDataMgr->oneHotEncodeLabel(pLabel, pOneHotLabel, 3);
            delete pLabel;
            m_pGNet->m_pLossLayer->setGroundTruth(*pOneHotLabel);
            delete pOneHotLabel;

            forwardG();
            backwardG();

            ++nIter;
        }
        m_pGNet->sgd(m_pGNet->getLearningRate(), i);
        ++batch;
    }
}

void Segmentation3DNet::trainD(){
    const long N = m_pDataMgr->m_NTrainFile;
    const int batchSize = m_pGNet->getBatchSize();
    long numBatch = N / batchSize;
    if (0 != N % batchSize) {
        numBatch += 1;
    }

    long nIter = 0;
    long batch = 0;
    vector<long> randSeq = generateRandomSequence(N);
    while (batch < numBatch) {
        m_pDNet->zeroParaGradient();
        int i = 0;
        int ignoreGx = 0;
        for (i = 0; i < batchSize && nIter < N; ++i) {
            long index = randSeq[nIter];

            Tensor<float> *pImage= nullptr;
            m_pDataMgr->readTrainImageFile(index, pImage);
            m_pGNet->m_pInputXLayer->setInputTensor(*pImage);
            m_pDNet->m_pInputXLayer->setInputTensor(*pImage);
            delete pImage;

            Tensor<unsigned char> *pLabel= nullptr;
            m_pDataMgr->readTrainLabelFile(index, pLabel);
            Tensor<float>* pOneHotLabel= nullptr;
            m_pDataMgr->oneHotEncodeLabel(pLabel, pOneHotLabel, 3);
            delete pLabel;
            m_pDNet->m_pGTLayer->setInputTensor(*pOneHotLabel);
            delete pOneHotLabel;

            // generative Gx
            forwardG();

            // train D for GT
            m_pDNet->setAlphaGroundTruth(true);
            switchDtoGT();
            forwardD();
            backwardD();

            // train D for Gx
            if (*m_pGNet->m_pGxLayer->m_pYTensor != *m_pDNet->m_pGTLayer->m_pYTensor){
                m_pDNet->setAlphaGroundTruth(false);
                switchDtoGx();
                forwardD();
                backwardD();
            }
            else{
                ++ignoreGx;
            }
            ++nIter;
        }
        m_pDNet->sgd(m_pDNet->getLearningRate(), i*2-ignoreGx);
        ++batch;
    }
}


float Segmentation3DNet::testG(){
    InputLayer *inputLayer = m_pGNet->m_pInputXLayer;
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) m_pGNet->getFinalLayer();
    long n = 0;
    float dice = 0.0;
    const long Ntest= 10000;
    while (n < Ntest) {
        //inputLayer->setInputTensor(m_pMnistData->m_pTestImages->slice(n));
        //lossLayer->setGroundTruth(constructGroundTruth(m_pMnistData->m_pTestLabels, n));
        m_pGNet->forwardPropagate();
        dice += lossLayer->diceCoefficient();
        ++n;
    }
    dice = dice/Ntest;
    cout<<"Info: average dice coefficient = "<<dice<<" in "<<Ntest<<" test samples."<<endl;
    return dice;
}

//construct a 2*1 one-hot vector
Tensor<float> Segmentation3DNet::constructGroundTruth(Tensor<unsigned char> *pLabels, const long index) {
    Tensor<float> tensor({10, 1});
    tensor.zeroInitialize();
    tensor.e(pLabels->e(index)) = 1;
    return tensor;
}


// pretrain an epoch for D
void Segmentation3DNet::pretrainD() {
    const long N = m_pDataMgr->m_NTrainFile;
    const int batchSize = m_pStubNet->getBatchSize();
    long numBatch = N / batchSize;
    if (0 != N % batchSize) {
        numBatch += 1;
    }

    long nIter = 0;
    long batch = 0;
    vector<long> randSeq = generateRandomSequence(N);
    while (batch < numBatch) {
        m_pDNet->zeroParaGradient();
        int i = 0;
        int ignoreStub = 0;
        for (i = 0; i < batchSize && nIter < N; ++i) {
            long index = randSeq[nIter];

            Tensor<float> *pImage= nullptr;
            m_pDataMgr->readTrainImageFile(index, pImage);
            m_pDNet->m_pInputXLayer->setInputTensor(*pImage);
            delete pImage;

            Tensor<unsigned char> *pLabel= nullptr;
            m_pDataMgr->readTrainLabelFile(index, pLabel);
            Tensor<float>* pOneHotLabel= nullptr;
            m_pDataMgr->oneHotEncodeLabel(pLabel, pOneHotLabel, 3);
            delete pLabel;
            m_pDNet->m_pGTLayer->setInputTensor(*pOneHotLabel);
            delete pOneHotLabel;

            m_pDNet->setAlphaGroundTruth(true);
            switchDtoGT();
            m_pDNet->forwardPropagate();
            m_pDNet->backwardPropagate(true);

            m_pStubNet->randomOutput();
            if (*m_pStubNet->getOutput() != *m_pDNet->m_pGTLayer->m_pYTensor){
                m_pDNet->setAlphaGroundTruth(false);
                switchDtoStub();
                m_pDNet->forwardPropagate();
                m_pDNet->backwardPropagate(true);
            }
            else {
                ++ignoreStub;
            }
            ++nIter;
        }
        m_pDNet->sgd(m_pDNet->getLearningRate(), i*2-ignoreStub);
        ++batch;
    }
}

void Segmentation3DNet::setDataMgr(DataManager* pDataMgr){
    m_pDataMgr = pDataMgr;
}

void Segmentation3DNet::setStubNet(StubNetForD* pStubNet){
    m_pStubNet = pStubNet;
}
