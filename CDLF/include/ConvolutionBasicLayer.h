//
// Created by Hui Xie on 11/19/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_CONVOLUTIONBASICLAYER_H
#define CDLF_FRAMEWORK_CONVOLUTIONBASICLAYER_H

#include "Layer.h"

class ConvolutionBasicLayer : public Layer {
public:
    ConvolutionBasicLayer(const int id, const string& name, Layer* prevLayer, const vector<long>& filterSize,
                          const int numFilters, const int stride);
    ~ConvolutionBasicLayer();

    Tensor<float>**  m_pW;
    Tensor<float>**  m_pdW;
    int m_numFilters;
    vector<long> m_filterSize;
    int m_stride;
    int m_OneFilterN;
    vector<long> m_tensorSizeBeforeCollapse;  //it does not include feature dimension, only for one filter

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void updateParameters(const float lr, const string& method, const int batchSize=1);

    bool checkFilterSize(const vector<long>& filterSize, Layer* prevLayer);
    void constructFiltersAndY();

    void computeOneFiterN();

    virtual  long getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);

protected:
    int m_NRange;  //N range for a thread to compute

};


#endif //CDLF_FRAMEWORK_CONVOLUTIONBASICLAYER_H
