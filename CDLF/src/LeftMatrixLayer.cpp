//
// Created by Hui Xie on 1/16/2019.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "statisTool.h"
#include <iostream>
#include <LeftMatrixLayer.h>
#include <TensorBlas.h>


using namespace std;

// y = W*x
// where y is m*n output matrix;
//       x is k*n input matrix;
//       W is m*k dimensional matrix
//       B is same size with y: m*n
LeftMatrixLayer::LeftMatrixLayer(const int id, const string& name, Layer* prevLayer, const vector<int>& filterSize)
        : MatrixLayer(id, name,prevLayer, filterSize, {filterSize[0], prevLayer->m_tensorSize[1]})
{
    m_type = "LeftMatrixLayer";
    if (prevLayer->m_tensorSize[0] != filterSize[1]){
        cout<<"Error: in LeftMatrixLayer, filterSize does not match previous Layer."<<endl;
    }
}

LeftMatrixLayer::~LeftMatrixLayer() {
   //null
}

void LeftMatrixLayer::forward() {
    *m_pYTensor = *m_pB;
    gemm(1.0, false, m_pW, false, m_prevLayer->m_pYTensor, 1, m_pYTensor);
}

//   y = W*x +B
//  dL/dW = dL/dy * dy/dW = dL/dy * x'
//  dL/dB = dL/dy * dy/dB = dL/dy
//  dL/dx = dL/dy * dy/dx = W' * dL/dy
void LeftMatrixLayer::backward(bool computeW, bool computeX) {
    Tensor<float> &dLdy = *m_pdYTensor;
    if (computeW){
        gemm(1.0, false, &dLdy, true, m_prevLayer->m_pYTensor, 1, m_pdW);
        matAdd(1, m_pdB, 1, &dLdy, m_pdB);
    }
    if (computeX){
        gemm(1.0, true, m_pW, false, &dLdy, 1, m_prevLayer->m_pdYTensor);
    }
}
