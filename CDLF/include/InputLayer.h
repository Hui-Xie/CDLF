//
// Created by Hui Xie on 6/6/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_INPUTLAYER_H
#define RL_NONCONVEX_INPUTLAYER_H

#include "Layer.h"



class InputLayer : public Layer{
public:
    InputLayer(const int id, const string& name,const vector<long>& tensorSize);
    ~InputLayer();

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward(bool computeW);
    virtual  void updateParameters(const float lr, const string& method, const int batchSize=1);
    void setInputTensor(const Tensor<float>& inputTensor);
    void setInputTensor(const Tensor<unsigned char>& inputTensor);
    virtual  long getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveArchitectLine(FILE* pFile);
};


#endif //RL_NONCONVEX_INPUTLAYER_H
