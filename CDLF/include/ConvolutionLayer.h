//
// Created by Hui Xie on 7/19/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_CONVOLUTIONLAYER_H
#define RL_NONCONVEX_CONVOLUTIONLAYER_H
#include "Layer.h"
#include <map>

/** Convolution layer
 * Y = W*X
 * where Y is the output at each voxel;
 *       W is the convolution filter, which is uniform in entire input;
 *       X is the receipt region of original input image;
 *       * indicate convolution
 *
 * Notes:
 * 1  in convolution layer, we do not consider bias, as there is a separate BiasLayer for use;
 * 2  Size changes: |Y| = (|X|-|W|)/stride + 1, in their different dimension;
 * 3  it is a good design if all numFilter is odd;
 * 4  Currently we supports 1D, 2D, 3D, 4D, 5D, 6D convolution; It is easy to extend to 7D or more.
 *
 * */


class ConvolutionLayer :  public Layer {
public:
    ConvolutionLayer(const int id, const string& name, Layer* prevLayer, const vector<long>& filterSize,
                     const int numFilters=1, const int stride=1);
    ~ConvolutionLayer();

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
    virtual  void saveArchitectLine(FILE* pFile);
};


#endif //RL_NONCONVEX_CONVOLUTIONLAYER_H
