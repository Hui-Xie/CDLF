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
            // cut original pLabel image to matched size
            Tensor<unsigned char>* pCutLabel = nullptr;
            vector<long> cutTensorSize = m_pDNet->m_pGTLayer->m_pYTensor->getDims();
            cutTensorSize.erase(cutTensorSize.begin());
            pLabel->subTensorFromTopLeft((pLabel->getDims()- cutTensorSize)/2,
                                         cutTensorSize, pCutLabel, 1);
            delete pLabel;

            Tensor<float>* pOneHotLabel= nullptr;
            m_pDataMgr->oneHotEncodeLabel(pCutLabel, pOneHotLabel, 3);
            delete pCutLabel;
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
            // cut original pLabel image to matched size
            Tensor<unsigned char>* pCutLabel = nullptr;
            vector<long> cutTensorSize = m_pGNet->m_pLossLayer->m_pYTensor->getDims();
            cutTensorSize.erase(cutTensorSize.begin());
            pLabel->subTensorFromTopLeft((pLabel->getDims()- cutTensorSize)/2,
                                         cutTensorSize, pCutLabel, 1);
            delete pLabel;
            Tensor<float>* pOneHotLabel= nullptr;
            m_pDataMgr->oneHotEncodeLabel(pCutLabel, pOneHotLabel, 3);
            delete pCutLabel;
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
            // cut original pLabel image to matched size
            Tensor<unsigned char>* pCutLabel = nullptr;
            vector<long> cutTensorSize = m_pDNet->m_pGTLayer->m_pYTensor->getDims();
            cutTensorSize.erase(cutTensorSize.begin());
            pLabel->subTensorFromTopLeft((pLabel->getDims()- cutTensorSize)/2,
                                         cutTensorSize, pCutLabel, 1);
            delete pLabel;
            Tensor<float>* pOneHotLabel= nullptr;
            m_pDataMgr->oneHotEncodeLabel(pCutLabel, pOneHotLabel, 3);
            delete pCutLabel;
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


float Segmentation3DNet::testG(bool outputFile){
    if (outputFile){
        if (!dirExist(m_pDataMgr->m_outputLabelsDir)){
            mkdir(m_pDataMgr->m_outputLabelsDir.c_str(),S_IRWXU |S_IRWXG | S_IROTH |S_IXOTH);
        }
    }

    const long N = m_pDataMgr->m_NTestFile;
    Tensor<float> dice({N,1});
    for (long i = 0; i <  N; ++i) {
        Tensor<float> *pImage = nullptr;
        m_pDataMgr->readTestImageFile(i, pImage);
        m_pGNet->m_pInputXLayer->setInputTensor(*pImage);
        delete pImage;

        Tensor<unsigned char> *pLabel = nullptr;
        m_pDataMgr->readTestLabelFile(i, pLabel);
        // cut original pLabel image to matched size
        Tensor<unsigned char>* pCutLabel = nullptr;
        vector<long> cutTensorSize = m_pGNet->m_pLossLayer->m_pYTensor->getDims();
        cutTensorSize.erase(cutTensorSize.begin());
        pLabel->subTensorFromTopLeft((pLabel->getDims()- cutTensorSize)/2,
                                     cutTensorSize, pCutLabel, 1);
        delete pLabel;

        Tensor<float> *pOneHotLabel = nullptr;
        m_pDataMgr->oneHotEncodeLabel(pCutLabel, pOneHotLabel, 3);
        delete pCutLabel;
        m_pGNet->m_pLossLayer->setGroundTruth(*pOneHotLabel);
        delete pOneHotLabel;

        forwardG();
        dice.e(i) = m_pGNet->m_pLossLayer->diceCoefficient();

        //0utput file
        if (outputFile){
            string filename = getFileName(m_pDataMgr->m_testImagesVector[i]);
            string outputFilename = m_pDataMgr->m_outputLabelsDir+ "/" + filename;
            m_pDataMgr->saveOneHotCode2LabelFile(m_pGNet->m_pGxLayer->m_pYTensor, outputFilename, m_pGNet->m_pInputXLayer->m_tensorSize);
         }
    }
    cout<<"Test "<<N<<" files: "<<"Dice avg=" <<dice.average()<<"; min="<<dice.min()
        <<"; max="<<dice.max()<<"; variance=" <<dice.variance() <<endl;

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
            // cut original pLabel image to matched size
            Tensor<unsigned char>* pCutLabel = nullptr;
            vector<long> cutTensorSize = m_pDNet->m_pGTLayer->m_pYTensor->getDims();
            cutTensorSize.erase(cutTensorSize.begin());
            pLabel->subTensorFromTopLeft((pLabel->getDims()- cutTensorSize)/2,
                                         cutTensorSize, pCutLabel, 1);
            delete pLabel;
            Tensor<float>* pOneHotLabel= nullptr;
            m_pDataMgr->oneHotEncodeLabel(pCutLabel, pOneHotLabel, 3);
            delete pCutLabel;
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
