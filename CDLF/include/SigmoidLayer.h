//
// Created by Hui Xie on 8/4/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_SIGMOIDLAYER_H
#define CDLF_FRAME_SIGMOIDLAYER_H

#include "Layer.h"

/* Y = k/( 1+ exp(-x)) in element-wise, where k is a constant integer
 * dL/dx = dL/dY * dY/dx = dL/dY * k* exp(x)/(1 +exp(x))^2
 * */

class SigmoidLayer : public Layer {
public:
    SigmoidLayer(const int id, const string& name,Layer* prevLayer, const vector<int>& tensorSize, const int k=1);
    ~SigmoidLayer();

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void averageParaGradient(const int batchSize);
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);
    virtual  void updateParameters(const float lr, const string& method, const int batchSize =1);

    virtual  void initializeLRs(const float lr);
    virtual  void updateLRs(const float deltaLoss, const int batchSize = 1);
    virtual  void updateParameters(const string& method, const int batchSize=1);

    virtual  int getNumParameters();
    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct();

private:
    int m_k;
};


#endif //CDLF_FRAME_SIGMOIDLAYER_H
