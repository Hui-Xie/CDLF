//
// Created by Hui Xie on 9/21/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_DATAMANAGER_H
#define CDLF_FRAMEWORK_DATAMANAGER_H

#include <string>
#include <vector>
#include "Tensor.h"
#include "ITKDataManager.h"

using namespace std;

class Seg3DDataManager : public ITKDataManager {
public:
    Seg3DDataManager(const string& dataSetDir);
    ~Seg3DDataManager();

    ITKImageIO<unsigned char, 3>*  m_labelItkImageIO;
    ITKImageIO<float, 3>*  m_imageItkImageIO;

    virtual void readImageFile(const string& filename, Tensor<float>*& pImage);
    virtual void readLabelFile(const string& filename, Tensor<float>*& pLabel);

    // the offset below should be Tensor Image dimension order, instead of ITK image dimension order
    virtual void saveLabel2File(Tensor<unsigned char>* pLabel, const vector<int>& offset, const string& fullPathFileName);
    virtual void freeLabelItkImageIO();
    virtual void freeImageItkImageIO();

};


#endif //CDLF_FRAMEWORK_DATAMANAGER_H
