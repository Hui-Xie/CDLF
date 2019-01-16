//
// Created by Hui Xie on 1/16/2019.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_MATRIXCONVERT_H
#define RL_NONCONVEX_MATRIXCONVERT_H

#include "Layer.h"
#include <map>


// y = W*x+B
// where y is m*n output matrix;
//       x is k*n input matrix;
//       W is m*k dimensional matrix
//       B is same size with y: m*n
class MatrixConvertLayer :  public Layer{
public:
    MatrixConvertLayer(const int id, const string& name, Layer* prevLayer, const vector<int>& filterSize);
    ~MatrixConvertLayer();

    Tensor<float>*  m_pW;
    Tensor<float>*  m_pB;
    Tensor<float>*  m_pdW;
    Tensor<float>*  m_pdB;


    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);
    virtual  void updateParameters(const float lr, const string& method, const int batchSize=1);


    virtual  int getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct(const int layerIndex);
};


#endif //RL_NONCONVEX_MATRIXCONVERT_H
