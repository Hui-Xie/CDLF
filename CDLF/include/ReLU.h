//
// Created by Hui Xie on 6/7/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_RELU_H
#define RL_NONCONVEX_RELU_H
#include "Layer.h"

class ReLU : public Layer {
public:
    ReLU(const int id, const string& name,Layer* prevLayer);
    ~ReLU();

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward(bool computeW);
    virtual  void updateParameters(const float lr, const string& method, const int batchSize =1);
    virtual  long getNumParameters();
    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveArchitectLine(FILE* pFile);
};


#endif //RL_NONCONVEX_RELU_H
