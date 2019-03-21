//
// Created by Hui Xie on 12/11/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAMEWORK_RESHAPELAYER_H
#define CDLF_FRAMEWORK_RESHAPELAYER_H

#include "Layer.h"

/*  Y = X.reshape();
 *  dL/dx = dL/dy
 * */

class ReshapeLayer : public Layer {
public:
    ReshapeLayer(const int id, const string& name, Layer* prevLayer, const vector<int>& outputSize);
    ~ReshapeLayer();

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



#endif //CDLF_FRAMEWORK_RESHAPELAYER_H
