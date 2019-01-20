//
// Created by Hui Xie on 1/19/2019.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "statisTool.h"
#include <iostream>
#include <RightMatrixLayer.h>
#include <TensorBlas.h>


using namespace std;

// Y = X*W +B
// where Y is m*n output matrix;
//       X is m*k input matrix;
//       W is k*n dimensional parameter matrix
//       B is same size with Y: m*n, Bias parameter matrix
RightMatrixLayer::RightMatrixLayer(const int id, const string& name, Layer* prevLayer, const vector<int>& filterSize)
        : MatrixLayer(id, name,prevLayer, filterSize, {prevLayer->m_tensorSize[0], filterSize[1]})
{
    m_type = "RightMatrixLayer";
    if (prevLayer->m_tensorSize[1] != filterSize[0]){
        cout<<"Error: in RightMatrixLayer, filterSize does not match previous Layer."<<endl;
    }
}

RightMatrixLayer::~RightMatrixLayer() {
    //null
}

// Y = X*W +B
void RightMatrixLayer::forward() {
    *m_pYTensor = *m_pB;
    gemm(1.0, false, m_prevLayer->m_pYTensor, false, m_pW, 1, m_pYTensor);
}

//   Y = X*W +B
//  dL/dW = dL/dy * dy/dW = X'* dL/dy
//  dL/dB = dL/dy * dy/dB = dL/dy
//  dL/dx = dL/dy * dy/dx = dL/dy *W'
void RightMatrixLayer::backward(bool computeW, bool computeX) {
    Tensor<float> &dLdy = *m_pdYTensor;
    if (computeW){
        gemm(1.0, true, m_prevLayer->m_pYTensor, false, &dLdy, 1, m_pdW);
        matAdd(1, m_pdB, 1, &dLdy, m_pdB);
    }
    if (computeX){
        gemm(1.0, false, &dLdy, true, m_pW, 1, m_prevLayer->m_pdYTensor);
    }
}
