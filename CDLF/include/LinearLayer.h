//
// Created by Hui Xie on 7/19/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_BIASLAYER_H
#define RL_NONCONVEX_BIASLAYER_H
#include "Layer.h"

/*  Element-wise Linear Layer
 *  It replaces old-version ScaleLayer and LinearLayer.
 *  Y_i = K_i*X_i + B_i  for each element
 *  where Y is the output of Linear Layer
 *        X is the input of the Linear Layer
 *        K is the learning parameter of Scale for each element
 *        B is the learning parameter of Bias for each element
 *  dL/dX = dL/dY * K_i    Where L is Loss
 *  dL/dk = dL/dY * X_i
 *  dL/db = dL/dY
 * */

class LinearLayer : public Layer {
public:
    LinearLayer(const int id, const string& name, Layer* prevLayer);
    ~LinearLayer();

    Tensor<float>*  m_pKTensor;
    Tensor<float>*  m_pBTensor;
    Tensor<float>*  m_pdKTensor;
    Tensor<float>*  m_pdBTensor;

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


#endif //RL_NONCONVEX_BIASLAYER_H
