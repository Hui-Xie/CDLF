//
// Created by Hui Xie on 1/5/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_HNDATAMANAGER_H
#define CDLF_FRAMEWORK_HNDATAMANAGER_H

#include "ITKDataManager.h"

class HNDataManager : public ITKDataManager {
public:
    HNDataManager(const string& dataSetDir);
    ~HNDataManager();

    // this ITKImageIO depends on the specific dataset type
    ITKImageIO<short, 3>*  m_labelItkImageIO;
    ITKImageIO<short, 3>*  m_imageItkImageIO;

    virtual void readImageFile(const string& filename, Tensor<float>*& pImage);
    virtual void readLabelFile(const string& filename, Tensor<float>*& pLabel);
    virtual void saveLabel2File(Tensor<unsigned char>* pLabel, const vector<int>& offset, const string& fullPathFileName);
    virtual void freeLabelItkImageIO();
    virtual void freeImageItkImageIO();


    string getLabelPathFrom(const string& imageFilePath);
    string generateLabelFilePath(const string& imageFilePath);
    vector<int>  getOutputOffset(const vector<int>& outputTensorSize);


};


#endif //CDLF_FRAMEWORK_HNDATAMANAGER_H
