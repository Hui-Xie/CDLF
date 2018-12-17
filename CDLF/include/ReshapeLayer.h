//
// Created by Hui Xie on 12/11/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

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
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);
    virtual  void updateParameters(const float lr, const string& method, const int batchSize =1);
    virtual  int getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct(const int layerIndex);
};



#endif //CDLF_FRAMEWORK_RESHAPELAYER_H
