//
// Created by Hui Xie on 9/14/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include <FileTools.h>
#include "Segmentation3DNet.h"



Segmentation3DNet::Segmentation3DNet(const string& name, GNet* pGNet, DNet* pDNet): GAN(name, pGNet, pDNet){

}

Segmentation3DNet::~Segmentation3DNet(){

}

void Segmentation3DNet::quicklySwitchTrainG_D(){
    const int N = m_pDataMgr->m_NTrainFile;
    const int batchSize = m_pGNet->getBatchSize();
    int numBatch = (N + batchSize -1)  / batchSize;
    int nIter = 0;
    int batch = 0;
    vector<int> randSeq = generateRandomSequence(N);
    bool bTrainSet = true;
    while (batch < numBatch) {
        m_pGNet->zeroParaGradient();
        m_pDNet->zeroParaGradient();
        int i = 0;
        int ignoreGx = 0;
        for (i = 0; i < batchSize && nIter < N; ++i) {
            int index = randSeq[nIter];

            Tensor<float> *pImage= nullptr;
            m_pDataMgr->readTrainImageFile(index, pImage);
            m_pGNet->m_pInputXLayer->setInputTensor(*pImage);
            m_pDNet->m_pInputXLayer->setInputTensor(*pImage);
            delete pImage;

            setOneHotLabel(bTrainSet, 3, index, m_pGNet->m_pLossLayer, m_pDNet->m_pGTLayer);

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
    const int N = m_pDataMgr->m_NTrainFile;
    const int batchSize = m_pGNet->getBatchSize();
    int numBatch = (N + batchSize -1) / batchSize;

    int nIter = 0;
    int batch = 0;
    vector<int> randSeq = generateRandomSequence(N);
    bool bTrainSet = true;

    m_pDNet->setAlphaGroundTruth(true);
    switchDtoGx();

    while (batch < numBatch) {
        m_pGNet->zeroParaGradient();
        int i = 0;
        for (i = 0; i < batchSize && nIter < N; ++i) {
            int index = randSeq[nIter];

            Tensor<float> *pImage= nullptr;
            m_pDataMgr->readTrainImageFile(index, pImage);
            m_pGNet->m_pInputXLayer->setInputTensor(*pImage);
            m_pDNet->m_pInputXLayer->setInputTensor(*pImage);
            delete pImage;

            setOneHotLabel(bTrainSet, 3, index, m_pGNet->m_pLossLayer, nullptr);

            forwardG();
            backwardG();

            ++nIter;
        }
        m_pGNet->sgd(m_pGNet->getLearningRate(), i);
        ++batch;
    }
}

void Segmentation3DNet::trainD(){
    const int N = m_pDataMgr->m_NTrainFile;
    const int batchSize = m_pGNet->getBatchSize();
    int numBatch = (N + batchSize -1)/ batchSize;
    int nIter = 0;
    int batch = 0;
    vector<int> randSeq = generateRandomSequence(N);
    bool bTrainSet = true;
    while (batch < numBatch) {
        m_pDNet->zeroParaGradient();
        int i = 0;
        int ignoreGx = 0;
        for (i = 0; i < batchSize && nIter < N; ++i) {
            int index = randSeq[nIter];

            Tensor<float> *pImage= nullptr;
            m_pDataMgr->readTrainImageFile(index, pImage);
            m_pGNet->m_pInputXLayer->setInputTensor(*pImage);
            m_pDNet->m_pInputXLayer->setInputTensor(*pImage);
            delete pImage;

            setOneHotLabel(bTrainSet, 3, index, nullptr, m_pDNet->m_pGTLayer);

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


float Segmentation3DNet::testG(bool outputFile){
    if (outputFile){
        if (!dirExist(m_pDataMgr->m_outputLabelsDir)){
            mkdir(m_pDataMgr->m_outputLabelsDir.c_str(),S_IRWXU |S_IRWXG | S_IROTH |S_IXOTH);
        }
    }
    bool bTrainSet = false;
    const int N = m_pDataMgr->m_NTestFile;
    Tensor<float> dice({N,1});
    Tensor<float> TPR({N,1});
    for (int i = 0; i <  N; ++i) {
        Tensor<float> *pImage = nullptr;
        m_pDataMgr->readTestImageFile(i, pImage);
        m_pGNet->m_pInputXLayer->setInputTensor(*pImage);
        delete pImage;

        setOneHotLabel(bTrainSet, 3, i, m_pGNet->m_pLossLayer, nullptr);

        m_pGNet->forwardPropagate();
        dice.e(i) = m_pGNet->m_pLossLayer->diceCoefficient();
        TPR.e(i) = m_pGNet->m_pLossLayer->getTPR();

        //0utput file
        if (outputFile){
            string filename = getFileName(m_pDataMgr->m_testImagesVector[i]);
            string outputFilename = m_pDataMgr->m_outputLabelsDir+ "/" + filename;
            m_pDataMgr->saveOneHotCode2LabelFile(m_pGNet->m_pGxLayer->m_pYTensor, outputFilename, m_pGNet->m_pInputXLayer->m_tensorSize);
         }
    }
    cout<<"Test "<<N<<" files: "<<endl;
    cout<<"Dice avg=" <<dice.average()<<"; min="<<dice.min() <<"; max="<<dice.max()<<"; variance=" <<dice.variance() <<endl;
    cout<<"TPR avg=" <<TPR.average()<<"; min="<<TPR.min() <<"; max="<<TPR.max()<<"; variance=" <<TPR.variance() <<endl;
}

// pretrain an epoch for D
void Segmentation3DNet::pretrainD() {
    const int N = m_pDataMgr->m_NTrainFile;
    const int batchSize = m_pStubNet->getBatchSize();
    int numBatch = (N + batchSize -1)/ batchSize;

    int nIter = 0;
    int batch = 0;
    vector<int> randSeq = generateRandomSequence(N);
    bool bTrainSet = true;
    while (batch < numBatch) {
        m_pDNet->zeroParaGradient();
        int i = 0;
        int ignoreStub = 0;
        for (i = 0; i < batchSize && nIter < N; ++i) {
            int index = randSeq[nIter];

            Tensor<float> *pImage= nullptr;
            m_pDataMgr->readTrainImageFile(index, pImage);
            m_pDNet->m_pInputXLayer->setInputTensor(*pImage);
            delete pImage;

            setOneHotLabel(bTrainSet, 3, i, nullptr, m_pDNet->m_pGTLayer);

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

void Segmentation3DNet::setDataMgr(Seg3DDataManager* pDataMgr){
    m_pDataMgr = pDataMgr;
}

void Segmentation3DNet::setStubNet(StubNetForD* pStubNet){
    m_pStubNet = pStubNet;
}

void Segmentation3DNet::setOneHotLabel(const bool bTrainSet, const int numLabels, const int indexImage,
                                       LossLayer *lossLayer, InputLayer *inputLayer) {
    Tensor<float> *pLabel = nullptr;
    if (bTrainSet) {
        m_pDataMgr->readTrainLabelFile(indexImage, pLabel);
    } else {
        m_pDataMgr->readTestLabelFile(indexImage, pLabel);
    }

    vector<int> cutTensorSize; //lossLayer and inputLayer should have same Tensor size
    if (nullptr != lossLayer){
        cutTensorSize = lossLayer->m_prevLayer->m_tensorSize;
    }
    if (nullptr != inputLayer){
        cutTensorSize = inputLayer->m_pYTensor->getDims();
    }
    cutTensorSize.erase(cutTensorSize.begin());

    Tensor<float> *pCutLabel = nullptr;
    if (cutTensorSize != pLabel->getDims()) {
        pCutLabel = new Tensor<float>(cutTensorSize);
        const vector<int> stride1 = vector<int>(cutTensorSize.size(),1);
        pLabel->subTensorFromTopLeft((pLabel->getDims() - cutTensorSize) / 2, pCutLabel, stride1);
        delete pLabel; pLabel = nullptr;
    } else {
        pCutLabel = pLabel;
    }

    Tensor<float> *pOneHotLabel = nullptr;
    m_pDataMgr->oneHotEncodeLabel(pCutLabel, pOneHotLabel, numLabels);
    delete pCutLabel; pCutLabel = nullptr;

    if (nullptr != lossLayer){
        lossLayer->setGroundTruth(*pOneHotLabel);
    }
    if (nullptr != inputLayer){
        inputLayer->setInputTensor(*pOneHotLabel);
    }
    delete pOneHotLabel; pOneHotLabel = nullptr;
}
