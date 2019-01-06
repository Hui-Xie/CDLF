//
// Created by Hui Xie on 1/5/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "HNDataManager.h"
#include <FileTools.h>

HNDataManager::HNDataManager(const string& dataSetDir) : ITKDataManager(dataSetDir) {
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

    m_labelItkImageIO = nullptr;
}

HNDataManager::~HNDataManager() {
    freeItkImageIO();
}


void HNDataManager::freeItkImageIO(){
    if (nullptr != m_labelItkImageIO){
        delete m_labelItkImageIO;
        m_labelItkImageIO = nullptr;
    }
}

void HNDataManager::readLabelFile(const string& filename, Tensor<float>*& pLabel){
    freeItkImageIO();
    m_labelItkImageIO = new ITKImageIO<short, 3>;
    Tensor<short>* pIOLabel = nullptr;
    m_labelItkImageIO->readFile(filename, pIOLabel);
    pLabel = new Tensor<float>;
    pLabel->valueTypeConvertFrom(*pIOLabel);
}


void HNDataManager::saveLabel2File(Tensor<unsigned char>* pLabel, const vector<int>& offset, const string& fullPathFileName){
    Tensor<short>* pIOLabel = new Tensor<short>;
    pIOLabel->valueTypeConvertFrom(*pLabel);
    m_labelItkImageIO->writeFileWithSameInputDim(pIOLabel, offset, fullPathFileName);
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
