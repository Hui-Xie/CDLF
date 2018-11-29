//
// Created by Hui Xie on 11/28/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_PADDINGLAYER_H
#define CDLF_FRAMEWORK_PADDINGLAYER_H

#include "Layer.h"

class PaddingLayer : public Layer {
public:

    PaddingLayer(const int id, const string &name, Layer *prevLayer, const vector<long>& tensorSize);

    ~PaddingLayer();

    virtual void initialize(const string &initialMethod);

    virtual void zeroParaGradient();

    virtual void forward();

    virtual void backward(bool computeW, bool computeX = true);

    virtual void updateParameters(const float lr, const string &method, const int batchSize = 1);
    virtual  long getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct(const int layerIndex);

private:
    vector<long> m_start;

};


#endif //CDLF_FRAMEWORK_PADDINGLAYER_H
