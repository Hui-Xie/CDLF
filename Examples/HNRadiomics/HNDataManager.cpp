//
// Created by Hui Xie on 1/5/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "HNDataManager.h"
#include <FileTools.h>

HNDataManager::HNDataManager(const string& dataSetDir) : ITKDataManager(dataSetDir) {
    m_labelItkImageIO = nullptr;
    m_imageItkImageIO = nullptr;

    if (!dataSetDir.empty()){
        // assign specific directories according to application
        m_trainImagesDir = m_dataSetDir +"/trainImages";
        m_trainLabelsDir = m_dataSetDir +"/trainLabels";
        m_testImagesDir = m_dataSetDir +"/testImages";
        m_testLabelsDir = m_dataSetDir +"/testLabels";
        m_outputLabelsDir = m_dataSetDir+"/OutputLabels";

        getFileVector(m_trainImagesDir, m_trainImagesVector);
        m_NTrainFile = m_trainImagesVector.size();
        cout<<"Info: totally read in "<<m_NTrainFile << " train images file. "<<endl;

        getFileVector(m_testImagesDir, m_testImagesVector);
        m_NTestFile = m_testImagesVector.size();
        cout<<"Info: totally read in "<<m_NTestFile << " test images file. "<<endl;
    }

}

HNDataManager::~HNDataManager() {
    freeLabelItkImageIO();
    freeImageItkImageIO();
}


void HNDataManager::freeLabelItkImageIO(){
    if (nullptr != m_labelItkImageIO){
        delete m_labelItkImageIO;
        m_labelItkImageIO = nullptr;
    }
}

void HNDataManager::freeImageItkImageIO(){
    if (nullptr != m_imageItkImageIO){
        delete m_imageItkImageIO;
        m_imageItkImageIO = nullptr;
    }
}

void HNDataManager::readImageFile(const string& filename, Tensor<float>*& pImage){
    freeImageItkImageIO();
    ITKImageIO<short, 3>* m_imageItkImageIO = new ITKImageIO<short, 3>;
    Tensor<short>* pShortImage = nullptr;
    m_imageItkImageIO->readFile(filename, pShortImage);
    pImage = new Tensor<float> ();
    pImage->valueTypeConvertFrom(*pShortImage);
    delete pShortImage;
}

void HNDataManager::readLabelFile(const string& filename, Tensor<float>*& pLabel){
    freeLabelItkImageIO();
    m_labelItkImageIO = new ITKImageIO<short, 3>;
    Tensor<short>* pIOLabel = nullptr;
    m_labelItkImageIO->readFile(filename, pIOLabel);
    pLabel = new Tensor<float>;
    pLabel->valueTypeConvertFrom(*pIOLabel);
    delete pIOLabel;
}


void HNDataManager::saveLabel2File(Tensor<unsigned char>* pLabel, const vector<int>& offset, const string& fullPathFileName){
    Tensor<short>* pIOLabel = new Tensor<short>;
    pIOLabel->valueTypeConvertFrom(*pLabel);
    m_labelItkImageIO->writeFileWithSameInputDim(pIOLabel, offset, fullPathFileName);
    delete pIOLabel;
}

string HNDataManager::getLabelPathFrom(const string &imagePath) {
    string imageFileName = getFileName(imagePath);
    size_t pos = imageFileName.find("_CT.nrrd");
    string labelFileName = imageFileName.replace(pos,string::npos, "_GTV.nrrd");

    if (string::npos != imagePath.find("trainImages")){
        labelFileName = m_trainLabelsDir +"/" +labelFileName;
    }
    else {
        labelFileName = m_testLabelsDir +"/" +labelFileName;
    }
    return labelFileName;
}
