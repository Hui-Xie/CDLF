//
// Created by Hui Xie on 7/19/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_CONVOLUTIONLAYER_H
#define CDLF_FRAME_CONVOLUTIONLAYER_H
#include "ConvolutionBasicLayer.h"


/** Convolution layer
 * Y = W*X + b
 * where Y is the output at each voxel;
 *       W is the convolution filter, which is uniform in entire input;
 *       X is the receipt region of original input image;
 *       b is the different bias for each filter
 *       * indicate convolution
 *
 * Notes:
 * 1  Size changes: |Y| = (|X|-|W|)/stride + 1, in their different dimension;
 * 2  it is a good design if all numFilter is odd;
 * 3  Currently we supports 1D, 2D, 3D, 4D, 5D, 6D convolution; It is easy to extend to 7D or more.
 *
 * */


class ConvolutionLayer :  public ConvolutionBasicLayer {
public:
    ConvolutionLayer(const int id, const string& name, Layer* prevLayer, const vector<int>& filterSize,
                     const vector<int>& stride, const int numFilters=1);
    ~ConvolutionLayer();


    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);
    void updateTensorSize();

private:
    void computeDW(const Tensor<float>* pdY, Tensor<float>* pdW);

    //Note: dx need to accumulate along filters
    // if pdx == nullptr, computeDx will use previousLayer->pdY;
    // if pdx !=  nullptr, computeX will use it to compute dx for one filter;
    void computeDX(const Tensor<float>* pExpandDY, const Tensor<float>* pW, Tensor<float>* pdX = nullptr);

};


#endif //CDLF_FRAME_CONVOLUTIONLAYER_H
