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

    // the offset below should be Tensor Image dimension order, instead of ITK image dimension order
    virtual void saveLabel2File(Tensor<unsigned char>* pLabel, const vector<int>& offset, const string& fullPathFileName);
    virtual void saveImage2File(Tensor<float>* pImage, const vector<int>& offset, const string& fullPathFileName);
    virtual void freeLabelItkImageIO();
    virtual void freeImageItkImageIO();


    virtual string getLabelPathFrom(const string& imageFilePath);
    string generateLabelFilePath(const string& imageFilePath);
    string generateFloatImagePath(const string& imageFilePath);

    // in Tensor dimension order
    vector<int>  getOutputOffset(const vector<int>& outputTensorSize);

    string getPatientCode(const string & imageFilename);


};


#endif //CDLF_FRAMEWORK_HNDATAMANAGER_H
