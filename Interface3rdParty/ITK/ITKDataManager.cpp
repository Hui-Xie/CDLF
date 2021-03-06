//
// Created by Hui Xie on 1/5/19.
// Copyright (c) 2019 Hui Xie. All rights reserved.

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

void ITKDataManager::saveOneHotCode2LabelFile(Tensor<float>* pOneHotLabel, const string& fullPathFileName, const vector<int>& offset){
    Tensor<unsigned char>* pLabel = nullptr;
    oneHot2Label(pOneHotLabel, pLabel);
    saveLabel2File(pLabel, offset, fullPathFileName);
    delete pLabel;
}

void ITKDataManager::generateLabelCenterMap() {
    cout<<"Info: generating label centers...."<<endl;
    m_mapTrainLabelCenter.clear();
    m_mapTestLabelCenter.clear();
    int N = m_trainImagesVector.size();
    for (int i= 0; i<N; ++i){
        Tensor<float>* pLabel = nullptr;
        const string labelFile = getLabelPathFrom(m_trainImagesVector[i]);
        readLabelFile(labelFile, pLabel);
        m_mapTrainLabelCenter[labelFile] = pLabel->getCenterOfNonZeroElements();
        delete pLabel;
    }
    N = m_testImagesVector.size();
    for (int i= 0; i<N; ++i){
        Tensor<float>* pLabel = nullptr;
        const string labelFile = getLabelPathFrom(m_testImagesVector[i]);
        readLabelFile(labelFile, pLabel);
        m_mapTestLabelCenter[labelFile] = pLabel->getCenterOfNonZeroElements();
        delete pLabel;
    }
    cout<<"Info: generating label centers ends."<<endl;
}



vector<int> ITKDataManager::getLabelCenter(const string labelFileName, const bool randomTranslation, const int translationMaxValue) {
    vector<int> center;

    if (m_mapTrainLabelCenter.count(labelFileName)){
        center = m_mapTrainLabelCenter[labelFileName];
    }
    else if (m_mapTestLabelCenter.count(labelFileName)){
        center = m_mapTestLabelCenter[labelFileName];
    }
    else{
        return center;
    }

    if (randomTranslation){
        randomTranslate(center, translationMaxValue);
    }
    return center;


}

