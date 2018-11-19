//
// Created by Hui Xie on 11/19/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_TRANSPOSEDCONVOLUTIONLAYER_H
#define CDLF_FRAMEWORK_TRANSPOSEDCONVOLUTIONLAYER_H

#include "ConvolutionBasicLayer.h"

class TransposedConvolutionLayer : public ConvolutionBasicLayer {
public:
    TransposedConvolutionLayer(const int id, const string& name, Layer* prevLayer, const vector<long>& filterSize,
                     const int numFilters, const int stride);
    ~TransposedConvolutionLayer();

    virtual  void forward();
    virtual  void backward(bool computeW);

    void updateTensorSize();

private:
    void expandDyTensor(const Tensor<float>* pdY, Tensor<float>* pExpandDY);
    void computeDW(const Tensor<float>* pdY, Tensor<float>* pdW);

    //Note: dx need to accumulate along filters
    // if pdx == nullptr, computeDx will use previousLayer->pdY;
    // if pdx !=  nullptr, computeX will use it to compute dx for one filter;
    void computeDX(const Tensor<float>* pExpandDY, const Tensor<float>* pW, Tensor<float>* pdX = nullptr);

    vector<long>  m_extendXTensorSize;
    int m_extendXStride;

};


#endif //CDLF_FRAMEWORK_TRANSPOSEDCONVOLUTIONLAYER_H
