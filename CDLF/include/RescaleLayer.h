//
// Created by Hui Xie on 12/17/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef RL_RESCALELAYER_H
#define RL_RESCALELAYER_H
#include "Layer.h"

/*
 *  Y = k*(X- Xmin)/(Xmax -Xmin);
 *  dL/dX = dL/dY * k/(Xmax -Xmin)
 *
 * */

class RescaleLayer : public Layer {
public:
    RescaleLayer(const int id, const string& name,Layer* prevLayer, float k = 1);
    ~RescaleLayer();

    float m_k;

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void averageParaGradient(const int batchSize);
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);

    virtual  void initializeLRs(const float lr);
    virtual  void updateLRs(const float deltaLoss);
    virtual  void updateParameters(Optimizer* pOptimizer);

    virtual  int getNumParameters();
    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct();
};


#endif //RL_RESCALELAYER_H
