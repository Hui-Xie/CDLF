//
// Created by Hui Xie on 1/16/2019.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_LEFTMATRIXCONVERT_H
#define RL_NONCONVEX_LEFTMATRIXCONVERT_H

#include "MatrixLayer.h"


// y = W*x+B
// where y is m*n output matrix;
//       x is k*n input matrix;
//       W is m*k dimensional matrix
//       B is same size with y: m*n
class LeftMatrixLayer :  public MatrixLayer{
public:
    LeftMatrixLayer(const int id, const string& name, Layer* prevLayer, const vector<int>& filterSize);
    ~LeftMatrixLayer();

    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);

};


#endif //RL_NONCONVEX_LEFTMATRIXCONVERT_H
