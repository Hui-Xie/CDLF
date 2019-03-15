//
// Created by Hui Xie on 1/16/2019.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_LEFTMATRIXCONVERT_H
#define CDLF_FRAME_LEFTMATRIXCONVERT_H

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


#endif //CDLF_FRAME_LEFTMATRIXCONVERT_H
