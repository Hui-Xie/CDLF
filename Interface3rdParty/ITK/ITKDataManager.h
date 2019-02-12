//
// Created by Hui Xie on 1/5/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_ITKDATAMANAGER_H
#define CDLF_FRAMEWORK_ITKDATAMANAGER_H

#include <string>
#include <vector>
#include "Tensor.h"
#include "ITKImageIO.h"

using namespace std;

class ITKDataManager {
public:
    ITKDataManager(const string& dataSetDir);
    ~ITKDataManager();

    string m_dataSetDir;
    string m_trainImagesDir;
    string m_trainLabelsDir;
    string m_testImagesDir;
    string m_testLabelsDir;

    string m_outputLabelsDir;

    vector<string> m_testImagesVector;
    vector<string> m_trainImagesVector;
    int m_NTrainFile;
    int m_NTestFile;

    // map<LabelFileName, labelCenter>
    map<string, vector<int>> m_mapTrainLabelCenter;
    map<string, vector<int>> m_mapTestLabelCenter;


    void readTrainImageFile(const int index, Tensor<float>*& pImage);
    void readTestImageFile(const int index, Tensor<float>*& pImage);
    void readTrainLabelFile(const int index, Tensor<float>*& pLabel);
    void readTestLabelFile(const int index, Tensor<float>*& pLabel);

    void oneHotEncodeLabel(const Tensor<float>* pLabel, Tensor<float>*& pOneHotLabel, const int k);
    void saveOneHotCode2LabelFile(Tensor<float>* pOneHotLabel, const string& fullPathFileName, const vector<int>& originalImageTensorSize);

    void oneHot2Label(Tensor<float>* pOneHotLabel,Tensor<unsigned char>*& pLabel);

    virtual void readImageFile(const string& filename, Tensor<float>*& pImage) = 0;
    virtual void readLabelFile(const string& filename, Tensor<float>*& pLabel)=0;
    virtual string getLabelPathFrom(const string& imageFilePath) = 0;

    // the offset below should be Tensor Image dimension order, instead of ITK image dimension order
    virtual void saveLabel2File(Tensor<unsigned char>* pLabel, const vector<int>& offset, const string& fullPathFileName) = 0;
    virtual void saveImage2File(Tensor<float>* pImage, const vector<int>& offset, const string& fullPathFileName) = 0;
    virtual void freeLabelItkImageIO() = 0;
    virtual void freeImageItkImageIO() = 0;

    void generateLabelCenterMap();
    vector<int> getLabelCenter(const string labelFileName, const bool randomTranslation, const int translationMaxValue);

    vector<int> getTopLeftIndexFrom(const vector<int>& imageDims, const vector<int>& subImageDims, const vector<int>& center);
};


#endif //CDLF_FRAMEWORK_ITKDATAMANAGER_H
