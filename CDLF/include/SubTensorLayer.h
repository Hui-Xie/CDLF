//
// Created by Hui Xie on 9/13/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef CDLF_FRAMEWORK_SUBTENSORLAYER_H
#define CDLF_FRAMEWORK_SUBTENSORLAYER_H

#include "Layer.h"

/* subTensorLayer extract a subTensor from previous layer.
 *
 * */

class SubTensorLayer : public Layer {
public:

    SubTensorLayer(const int id, const string &name, Layer *prevLayer, const vector<int>& start, const vector<int>& span);

    ~SubTensorLayer();

    virtual void initialize(const string &initialMethod);

    virtual void zeroParaGradient();

    virtual void forward();

    virtual void backward(bool computeW, bool computeX = true);

    virtual void updateParameters(const float lr, const string &method, const int batchSize = 1);
    virtual  int getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct(const int layerIndex);

private:
    vector<int> m_start;
    vector<int> m_span;

};




#endif //CDLF_FRAMEWORK_SUBTENSORLAYER_H
