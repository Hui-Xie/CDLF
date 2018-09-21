//
// Created by Hui Xie on 9/21/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "DataManager.h"
#include "FileTools.h"
#include "ITKImageIO.h"


DataManager::DataManager(const string dataSetDir) {
    m_dataSetDir = dataSetDir;
    m_trainImagesDir = m_dataSetDir +"/trainImages";
    m_trainLabelsDir = m_dataSetDir +"/trainLabels";
    m_testImagesDir = m_dataSetDir +"/testImages";
    m_testLabelsDir = m_dataSetDir +"/testLabels";

    getFileVector(m_trainImagesDir, m_trainImagesVector);
    m_NTrainFile = m_trainImagesDir.size();
    cout<<"Info: totally read in "<<m_NTrainFile << "train images file. "<<endl;

    getFileVector(m_testImagesDir, m_testImagesVector);
    m_NTestFile = m_testImagesDir.size();
    cout<<"Info: totally read in "<<m_NTestFile << "test images file. "<<endl;

}

DataManager::~DataManager(){

}

void DataManager::readTrainImageFile(const int index, Tensor<float>* pImage){
     readImageFile(m_trainImagesVector[index], pImage);
}

void DataManager::readTestImageFile(const int index, Tensor<float>* pImage){
    readImageFile(m_testImagesVector[index], pImage);
}

void DataManager::readTrainLabelFile(const int index, Tensor<float>* pLabel){
    string filename = getFileName(m_trainImagesVector[index]);
    readLabelFile(m_trainLabelsDir+"/"+filename, pLabel);
}

void DataManager::readTestLabelFile(const int index, Tensor<float>* pLabel){
    string filename = getFileName(m_testImagesVector[index]);
    readLabelFile(m_testLabelsDir+"/"+filename, pLabel);
}

void DataManager::readImageFile(const string& filename, Tensor<float>* pImage){
    ITKImageIO<float, 3> itkImageIO;
    itkImageIO.readFile(filename, pImage);
}

void DataManager::readLabelFile(const string& filename, Tensor<float>* pLabel){
    ITKImageIO<unsigned char, 3> itkImageIO;
    itkImageIO.readFile(filename, pLabel);
}

// k indicates number of categories
// the original label must be continous integer starting from 0.
void DataManager::oneHotEncodeLabel(const Tensor<float>* pLabel, Tensor<float>* pOneHotLabel, const int k){
    const long N = pLabel->getLength();
    vector<long> newDims =  pLabel->getDims();
    newDims.insert(newDims.begin(), k);
    pOneHotLabel = new Tensor<float> (newDims);
    pOneHotLabel->zeroInitialize();
    for(long i=0; i<N; ++i){
        int label = int(pLabel->e(i));
        pOneHotLabel->e((label%k)*N+i) = 1.0;
    }
}