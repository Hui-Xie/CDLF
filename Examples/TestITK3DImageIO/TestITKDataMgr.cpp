//
// Created by Hui Xie on 1/14/19.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#include "TestITKDataMgr.h"
#include <FileTools.h>

TestITKDataMgr::TestITKDataMgr(const string& dataSetDir) : ITKDataManager(dataSetDir) {
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

TestITKDataMgr::~TestITKDataMgr() {
    freeLabelItkImageIO();
    freeImageItkImageIO();
}


void TestITKDataMgr::freeLabelItkImageIO(){
    if (nullptr != m_labelItkImageIO){
        delete m_labelItkImageIO;
        m_labelItkImageIO = nullptr;
    }
}

void TestITKDataMgr::freeImageItkImageIO(){
    if (nullptr != m_imageItkImageIO){
        delete m_imageItkImageIO;
        m_imageItkImageIO = nullptr;
    }
}

void TestITKDataMgr::readImageFile(const string& filename, Tensor<float>*& pImage){
    freeImageItkImageIO();
    m_imageItkImageIO = new ITKImageIO<short, 3>;
    Tensor<short>* pShortImage = nullptr;
    m_imageItkImageIO->readFile(filename, pShortImage);
    pImage = new Tensor<float> ();
    pImage->valueTypeConvertFrom(*pShortImage);
    delete pShortImage;
}

void TestITKDataMgr::readLabelFile(const string& filename, Tensor<float>*& pLabel){
    freeLabelItkImageIO();
    m_labelItkImageIO = new ITKImageIO<short, 3>;
    Tensor<short>* pIOLabel = nullptr;
    m_labelItkImageIO->readFile(filename, pIOLabel);
    pLabel = new Tensor<float>;
    pLabel->valueTypeConvertFrom(*pIOLabel);
    delete pIOLabel;
}


void TestITKDataMgr::saveLabel2File(Tensor<unsigned char>* pLabel, const vector<int>& offset, const string& fullPathFileName){
    Tensor<short>* pIOLabel = new Tensor<short>;
    pIOLabel->valueTypeConvertFrom(*pLabel);
    vector<int> reverseOffset = reverseVector(offset); // very important
    if (nullptr != m_labelItkImageIO){
        m_labelItkImageIO->writeFileWithSameInputDim(pIOLabel, reverseOffset, fullPathFileName);
    }
    else{
        m_imageItkImageIO->writeFileWithSameInputDim(pIOLabel, reverseOffset, fullPathFileName);
    }
    delete pIOLabel;
}

string TestITKDataMgr::getLabelPathFrom(const string &imageFilePath) {
    if (m_trainLabelsDir.empty()){
        return "";
    }
    else{
        string imageFileName = getFileName(imageFilePath);
        size_t pos = imageFileName.find("_CT.nrrd");
        string labelFileName = imageFileName.replace(pos,string::npos, "_GTV.nrrd");

        if (string::npos != imageFilePath.find("trainImages")){
            labelFileName = m_trainLabelsDir +"/" +labelFileName;
        }
        else {
            labelFileName = m_testLabelsDir +"/" +labelFileName;
        }
        return labelFileName;
    }
}

string TestITKDataMgr::generateLabelFilePath(const string &imageFilePath) {
    string imageFile = imageFilePath;
    size_t pos = imageFile.find(".nrrd");
    string labelFilePath = imageFile.replace(pos,string::npos, "_Label.nrrd");
    return labelFilePath;
}

vector<int> TestITKDataMgr::getOutputOffset(const vector<int> &outputTensorSize, const vector<int> & center) {
    if (nullptr != m_labelItkImageIO){
        return  m_labelItkImageIO->getOutputOffset(outputTensorSize, center);
    }
    else{
        return m_imageItkImageIO->getOutputOffset(outputTensorSize, center);
    }
}

void TestITKDataMgr::saveImage2File(Tensor<float> *pImage, const vector<int> &offset, const string &fullPathFileName) {
    ITKImageIO<float, 3> floatItkImageIO;
    floatItkImageIO.copyImagePropertyFrom(*m_imageItkImageIO);
    vector<int> reverseOffset = reverseVector(offset);
    floatItkImageIO.writeFileWithSameInputDim(pImage, reverseOffset, fullPathFileName);
}


