//
// Created by Hui Xie on 6/12/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_NORMALIZATIONLAYER_H
#define RL_NONCONVEX_NORMALIZATIONLAYER_H
#include "Layer.h"

// NormalizationLayer only has one previous layer.

class NormalizationLayer: public Layer {
public:
    NormalizationLayer(const int id, const string& name,Layer* prevLayer);
    ~NormalizationLayer();
    float m_epsilon;

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);
    virtual  void updateParameters(const float lr, const string& method, const int batchSize =1);
    virtual  int getNumParameters();
    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct(const int layerIndex);

};


#endif //RL_NONCONVEX_NORMALIZATIONLAYER_H
