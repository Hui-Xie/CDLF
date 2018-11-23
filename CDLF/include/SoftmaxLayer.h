//
// Created by Hui Xie on 7/28/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_SOFTMAXLAYER_H
#define RL_NONCONVEX_SOFTMAXLAYER_H

#include "Layer.h"

/* SoftMax Layer only suit for mutually exclusive output classification in each data point
 * This softmaxLayer support N-d tensor. It implements softmax on the 0th dimension of the N-d tensor.
 * Y_i = exp(X_i)/ (\sum exp(x_j))
 * dL/dx_i = \sum(dL/dy_i * dy_i/dx_i) = exp(x_i)*{dL/dy_i * \sum exp(x_j)-\sum(dL/dy_j*exp(x_j)}/(\sum exp(x_j))^2
 *  where j is the range variable between 1 to N.
 *
 *
 * */

class SoftmaxLayer : public Layer {
public:
    SoftmaxLayer(const int id, const string& name,Layer* prevLayer);
    ~SoftmaxLayer();

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward(bool computeW);
    virtual  void updateParameters(const float lr, const string& method, const int batchSize=1);
    virtual  long getNumParameters();
    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct(const int layerIndex);

};

#endif //RL_NONCONVEX_SOFTMAXLAYER_H
