//
// Created by Hui Xie on 9/10/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAMEWORK_EXPONENTIALLAYER_H
#define CDLF_FRAMEWORK_EXPONENTIALLAYER_H

#include "Layer.h"

/* y_i = exp(x_i)
 * */

class ExponentialLayer : public Layer {
public:
    ExponentialLayer(const int id, const string& name,Layer* prevLayer);
    ~ExponentialLayer();

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void averageParaGradient(const int batchSize);
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);
    virtual  void updateParameters(const float lr, const string& method);
    virtual  int getNumParameters();

    virtual  void initializeLRs(const float lr);
    virtual  void updateLRs(const float deltaLoss);
    virtual  void updateParameters(const string& method);

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct();
};



#endif //CDLF_FRAMEWORK_EXPONENTIALLAYER_H
