//
// Created by Hui Xie on 6/5/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_FCLAYER_H
#define RL_NONCONVEX_FCLAYER_H

#include "Layer.h"
#include <map>


// y = W*x+ b
// where y and b is m-D vector, y is output vector;
//       x is n-D input vector
//       W is m*n dimensional matrix
class FCLayer :  public Layer{
public:
     FCLayer(const int id, const string& name, Layer* prevLayer, const long outputWidth);
    ~FCLayer();

    long m_m; //output width
    long m_n; //input width
    Tensor<float>*  m_pW;
    Tensor<float>*  m_pBTensor;
    Tensor<float>*  m_pdW;
    Tensor<float>*  m_pdBTensor;

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward(bool computeW);
    virtual  void updateParameters(const float lr, const string& method, const int batchSize=1);


    void printWandBVector();
    void printdWanddBVector();
    virtual  long getNumParameters();
};


#endif //RL_NONCONVEX_FCLAYER_H
