//
// Created by Hui Xie on 9/13/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

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
    virtual  void averageParaGradient(const int batchSize);
    virtual void forward();
    virtual void backward(bool computeW, bool computeX = true);

    virtual  void initializeLRs(const float lr);
    virtual  void updateLRs(const float deltaLoss);
    virtual  void updateParameters(const string& method, Optimizer* pOptimizer);


    virtual  int getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct();

private:
    vector<int> m_start;
    vector<int> m_span;

};




#endif //CDLF_FRAMEWORK_SUBTENSORLAYER_H
