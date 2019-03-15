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

    virtual void forward();

    virtual void backward(bool computeW, bool computeX = true);

    virtual void updateParameters(const float lr, const string &method, const int batchSize = 1);
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
