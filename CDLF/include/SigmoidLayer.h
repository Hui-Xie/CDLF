//
// Created by Hui Xie on 8/4/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_SIGMOIDLAYER_H
#define RL_NONCONVEX_SIGMOIDLAYER_H

#include "Layer.h"

/* Y = k/( 1+ exp(-x)) in element-wise, where k is a constant integer
 * dL/dx = dL/dY * dY/dx = dL/dY * k* exp(x)/(1 +exp(x))^2
 * */

class SigmoidLayer : public Layer {
public:
    SigmoidLayer(const int id, const string& name,Layer* prevLayer, const int k=1);
    ~SigmoidLayer();

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward(bool computeW);
    virtual  void updateParameters(const float lr, const string& method, const int batchSize =1);
    virtual  long getNumParameters();
    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveArchitectLine(FILE* pFile);

private:
    int m_k;
};


#endif //RL_NONCONVEX_SIGMOIDLAYER_H
