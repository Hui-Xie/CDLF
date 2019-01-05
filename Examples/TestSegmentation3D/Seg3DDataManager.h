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


    virtual void readLabelFile(const string& filename, Tensor<float>*& pLabel);
    virtual void saveLabel2File(Tensor<unsigned char>* pLabel, const vector<int>& offset, const string& fullPathFileName);
    virtual void freeItkImageIO();

};


#endif //CDLF_FRAMEWORK_DATAMANAGER_H
