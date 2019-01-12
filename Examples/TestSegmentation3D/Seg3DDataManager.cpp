//
// Created by Hui Xie on 9/21/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "Seg3DDataManager.h"
#include "FileTools.h"



Seg3DDataManager::Seg3DDataManager(const string& dataSetDir) : ITKDataManager(dataSetDir) {
    m_labelItkImageIO = nullptr;
    m_imageItkImageIO = nullptr;

    if (!dataSetDir.empty()) {
        m_trainImagesDir = m_dataSetDir + "/trainImages";
        m_trainLabelsDir = m_dataSetDir + "/trainLabels";
        m_testImagesDir = m_dataSetDir + "/testImages";
        m_testLabelsDir = m_dataSetDir + "/testLabels";
        m_outputLabelsDir = m_dataSetDir + "/OutputLabels";

        getFileVector(m_trainImagesDir, m_trainImagesVector);
        m_NTrainFile = m_trainImagesVector.size();
        cout << "Info: totally read in " << m_NTrainFile << " train images file. " << endl;

        getFileVector(m_testImagesDir, m_testImagesVector);
        m_NTestFile = m_testImagesVector.size();
        cout << "Info: totally read in " << m_NTestFile << " test images file. " << endl;
    }
}

Seg3DDataManager::~Seg3DDataManager(){
    freeLabelItkImageIO();
    freeImageItkImageIO();
}

void Seg3DDataManager::freeLabelItkImageIO(){
    if (nullptr != m_labelItkImageIO){
        delete m_labelItkImageIO;
        m_labelItkImageIO = nullptr;
    }
}

void Seg3DDataManager::freeImageItkImageIO(){
    if (nullptr != m_imageItkImageIO){
        delete m_imageItkImageIO;
        m_imageItkImageIO = nullptr;
    }
}

void Seg3DDataManager::readImageFile(const string& filename, Tensor<float>*& pImage){
    freeImageItkImageIO();
    m_imageItkImageIO = new ITKImageIO<float, 3>;
    m_imageItkImageIO->readFile(filename, pImage);
}


void Seg3DDataManager::readLabelFile(const string& filename, Tensor<float>*& pLabel){
    freeLabelItkImageIO();
    m_labelItkImageIO = new ITKImageIO<unsigned char, 3>;
    Tensor<unsigned char>* pIOLabel = nullptr;
    m_labelItkImageIO->readFile(filename, pIOLabel);
    pLabel = new Tensor<float>;
    pLabel->valueTypeConvertFrom(*pIOLabel);
    delete pIOLabel;
}


void Seg3DDataManager::saveLabel2File(Tensor<unsigned char>* pLabel, const vector<int>& offset, const string& fullPathFileName){
    if (nullptr != m_labelItkImageIO){
        m_labelItkImageIO->writeFileWithSameInputDim(pLabel, offset, fullPathFileName);
    }
    else{
        m_imageItkImageIO->writeFileWithSameInputDim(pLabel, offset, fullPathFileName);
    }
}


