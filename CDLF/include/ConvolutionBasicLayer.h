//
// Created by Hui Xie on 11/19/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_CONVOLUTIONBASICLAYER_H
#define CDLF_FRAMEWORK_CONVOLUTIONBASICLAYER_H

#include "Layer.h"

class ConvolutionBasicLayer : public Layer {
public:
    ConvolutionBasicLayer(const int id, const string& name, Layer* prevLayer, const vector<int>& filterSize,
                          const vector<int>& stride, const int numFilters);
    ~ConvolutionBasicLayer();

    Tensor<float>**  m_pW;
    Tensor<float>**  m_pdW;
    int m_numFilters;
    vector<int> m_filterSize;
    vector<int> m_feature_filterSize;  //include feature dimension, even 1.
    int m_numInputFeatures;
    int m_numOutputFeatures;


    vector<int> m_stride;
    vector<int> m_feature_stride;
    int m_OneFilterN;
    vector<int> m_tensorSizeBeforeCollapse;  //it does not include feature dimension, only for one filter

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void updateParameters(const float lr, const string& method, const int batchSize=1);

    bool checkFilterSize(Layer* prevLayer, const vector<int>& filterSize, const vector<int>& stride, const int numFilters);
    void constructFiltersAndY();

    void computeOneFiterN();

    virtual  int getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct();

protected:
    void updateFeatureFilterSize();


};


#endif //CDLF_FRAMEWORK_CONVOLUTIONBASICLAYER_H
