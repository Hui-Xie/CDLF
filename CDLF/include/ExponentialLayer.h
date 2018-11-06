//
// Created by Hui Xie on 9/10/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

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
    virtual  void forward();
    virtual  void backward(bool computeW);
    virtual  void updateParameters(const float lr, const string& method, const int batchSize =1);
    virtual  long getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
};



#endif //CDLF_FRAMEWORK_EXPONENTIALLAYER_H
