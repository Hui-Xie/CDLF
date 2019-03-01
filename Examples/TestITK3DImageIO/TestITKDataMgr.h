//
// Created by Hui Xie on 1/14/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_TESTITKDATAMGR_H
#define CDLF_FRAMEWORK_TESTITKDATAMGR_H

#include "ITKDataManager.h"


class TestITKDataMgr  : public ITKDataManager {
public:
    TestITKDataMgr(const string& dataSetDir);
    ~TestITKDataMgr();

    // this ITKImageIO depends on the specific dataset type
    ITKImageIO<short, 3>*  m_labelItkImageIO;
    ITKImageIO<short, 3>*  m_imageItkImageIO;

    virtual void readImageFile(const string& filename, Tensor<float>*& pImage);
    virtual void readLabelFile(const string& filename, Tensor<float>*& pLabel);

    // the offset below should be Tensor Image dimension order, instead of ITK image dimension order
    virtual void saveLabel2File(Tensor<unsigned char>* pLabel, const vector<int>& offset, const string& fullPathFileName);
    virtual void saveImage2File(Tensor<float>* pImage, const vector<int>& offset, const string& fullPathFileName);
    virtual void freeLabelItkImageIO();
    virtual void freeImageItkImageIO();


    string getLabelPathFrom(const string& imageFilePath);
    string generateLabelFilePath(const string& imageFilePath);
    vector<int>  getOutputOffset(const vector<int>& outputTensorSize, const vector<int>& center);


};


#endif //CDLF_FRAMEWORK_TESTITKDATAMGR_H
