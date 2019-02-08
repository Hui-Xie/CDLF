//
// Created by Hui Xie on 1/5/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "ITKDataManager.h"
#include <FileTools.h>


ITKDataManager::ITKDataManager(const string& dataSetDir) {
    m_dataSetDir = dataSetDir;
    m_trainImagesDir = "";
    m_trainLabelsDir = "";
    m_testImagesDir = "";
    m_testLabelsDir = "";

    m_outputLabelsDir = "";
    m_NTrainFile = 0;
    m_NTestFile = 0;

}

ITKDataManager::~ITKDataManager(){

}


void ITKDataManager::readTrainImageFile(const int index, Tensor<float>*& pImage){
    readImageFile(m_trainImagesVector[index], pImage);
}

void ITKDataManager::readTestImageFile(const int index, Tensor<float>*& pImage){
    readImageFile(m_testImagesVector[index], pImage);
}

void ITKDataManager::readTrainLabelFile(const int index, Tensor<float>*& pLabel){
    string filename = getFileName(m_trainImagesVector[index]);
    readLabelFile(m_trainLabelsDir+"/"+filename, pLabel);
}

void ITKDataManager::readTestLabelFile(const int index, Tensor<float>*& pLabel){
    string filename = getFileName(m_testImagesVector[index]);
    readLabelFile(m_testLabelsDir+"/"+filename, pLabel);
}

// k indicates number of categories
// the original label must be continuous integer starting from 0.
void ITKDataManager::oneHotEncodeLabel(const Tensor<float>* pLabel, Tensor<float>*& pOneHotLabel, const int k){
    const int N = pLabel->getLength();
    vector<int> newDims =  pLabel->getDims();
    newDims.insert(newDims.begin(), k);
    pOneHotLabel = new Tensor<float> (newDims);
    pOneHotLabel->zeroInitialize();
    for(int i=0; i<N; ++i){
        int label = int(pLabel->e(i));
        pOneHotLabel->e((label%k)*N+i) = 1.0;
    }
}

void ITKDataManager::oneHot2Label(Tensor<float>* pOneHotLabel, Tensor<unsigned char>*& pLabel){
    if (nullptr == pLabel){
        pLabel = new Tensor<unsigned char>();
    }
    *pLabel= pOneHotLabel->getMaxPositionSubTensor();
}

void ITKDataManager::saveOneHotCode2LabelFile(Tensor<float>* pOneHotLabel, const string& fullPathFileName, const vector<int>& originalImageTensorSize){
    Tensor<unsigned char>* pLabel = nullptr;
    oneHot2Label(pOneHotLabel, pLabel);
    vector<int> offset = (originalImageTensorSize - pLabel->getDims())/2;
    saveLabel2File(pLabel, offset, fullPathFileName);
    delete pLabel;
}

void ITKDataManager::generateLabelCenterMap() {
    m_mapTrainLabelCenter.clear();
    m_mapTestLabelCenter.clear();
    int N = m_trainImagesVector.size();
    for (int i= 0; i<N; ++i){
        Tensor<float>* pLabel = nullptr;
        readLabelFile(m_trainImagesVector[i], pLabel);
        m_mapTrainLabelCenter[m_trainImagesVector[i]] = pLabel->getCenterOfNonZeroElements();
        delete pLabel;
    }
    N = m_testImagesVector.size();
    for (int i= 0; i<N; ++i){
        Tensor<float>* pLabel = nullptr;
        readLabelFile(m_testImagesVector[i], pLabel);
        m_mapTestLabelCenter[m_testImagesVector[i]] = pLabel->getCenterOfNonZeroElements();
        delete pLabel;
    }
}

vector<int> ITKDataManager::getTopLeftIndexFrom(const vector<int> &imageDims, const vector<int> &subImageDims,
                                                const vector<int>&  center) {
    assert(subImageDims <= imageDims);
    assert(center >= subImageDims/2);

    if (center.empty()){
        return (imageDims -subImageDims)/2;
    }
    else {
        return (center - subImageDims/2);
    }
}

vector<int> ITKDataManager::getLabelCenter(const string labelFileName) {
    if (m_mapTrainLabelCenter.count(labelFileName)){
        return m_mapTrainLabelCenter[labelFileName];
    }
    else if (m_mapTestLabelCenter.count(labelFileName)){
        return m_mapTestLabelCenter[labelFileName];
    }
    else{
        return vector<int>();
    }
}

