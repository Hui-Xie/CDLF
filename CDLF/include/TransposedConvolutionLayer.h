//
// Created by Hui Xie on 11/19/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_TRANSPOSEDCONVOLUTIONLAYER_H
#define CDLF_FRAMEWORK_TRANSPOSEDCONVOLUTIONLAYER_H

#include "Layer.h"

class TransposedConvolutionLayer : public Layer {
public:
    TransposedConvolutionLayer(const int id, const string& name, Layer* prevLayer, const vector<long>& filterSize,
                     const int numFilters, const int stride);
    ~TransposedConvolutionLayer();

    Tensor<float>**  m_pW;
    Tensor<float>**  m_pdW;
    int m_numFilters;
    vector<long> m_filterSize;

    void constructFiltersAndY();


    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward(bool computeW);
    virtual  void updateParameters(const float lr, const string& method, const int batchSize=1);

private:
    int m_stride;
    int m_OneFilterN;

    bool checkFilterSize(const vector<long>& filterSize, Layer* prevLayer);
    void expandDyTensor(const Tensor<float>* pdY, Tensor<float>* pExpandDY);
    void computeDW(const Tensor<float>* pdY, Tensor<float>* pdW);

    //Note: dx need to accumulate along filters
    // if pdx == nullptr, computeDx will use previousLayer->pdY;
    // if pdx !=  nullptr, computeX will use it to compute dx for one filter;
    void computeDX(const Tensor<float>* pExpandDY, const Tensor<float>* pW, Tensor<float>* pdX = nullptr);
    void updateTensorSize();
    void computeOneFiterN();

    virtual  long getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
};


#endif //CDLF_FRAMEWORK_TRANSPOSEDCONVOLUTIONLAYER_H
