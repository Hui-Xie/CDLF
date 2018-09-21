//
// Created by Hui Xie on 9/21/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_DATAMANAGER_H
#define CDLF_FRAMEWORK_DATAMANAGER_H

#include <string>
#include <vector>
#include "Tensor.h"

using namespace std;

class DataManager {
public:
    DataManager(const string dataSetDir);
    ~DataManager();

    string m_dataSetDir;
    string m_trainImagesDir;
    string m_trainLabelsDir;
    string m_testImagesDir;
    string m_testLabelsDir;

    vector<string> m_testImagesVector;
    vector<string> m_trainImagesVector;
    int m_NTrainFile;
    int m_NTestFile;


    void readTrainImageFile(const int index, Tensor<float>* pImage);
    void readTestImageFile(const int index, Tensor<float>* pImage);
    void readTrainLabelFile(const int index, Tensor<unsigned char>* pLabel);
    void readTestLabelFile(const int index, Tensor<unsigned char>* pLabel);


private:
    void readImageFile(const string& filename, Tensor<float>* pImage);
    void readLabelFile(const string& filename, Tensor<unsigned char>* pLabel);

};


#endif //CDLF_FRAMEWORK_DATAMANAGER_H
