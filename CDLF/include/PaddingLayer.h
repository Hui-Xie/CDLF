//
// Created by Hui Xie on 11/28/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_PADDINGLAYER_H
#define CDLF_FRAMEWORK_PADDINGLAYER_H

#include "Layer.h"

class PaddingLayer : public Layer {
public:

    PaddingLayer(const int id, const string &name, Layer *prevLayer, const vector<int>& tensorSize, const float initialValue);

    ~PaddingLayer();

    virtual void initialize(const string &initialMethod);

    virtual void zeroParaGradient();
    virtual  void averageParaGradient(const int batchSize);
    virtual void forward();
    virtual void backward(bool computeW, bool computeX = true);

    //virtual  void initializeLRs(const float lr);
    //virtual  void updateLRs(const float deltaLoss);
    virtual  void updateParameters(const string& method, Optimizer* pOptimizer);

    virtual  int getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct();

private:
    vector<int> m_start;
    float m_initialValue;

};


#endif //CDLF_FRAMEWORK_PADDINGLAYER_H
