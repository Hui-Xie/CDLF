//
// Created by Hui Xie on 8/4/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_BRANCHLAYER_H
#define CDLF_FRAME_BRANCHLAYER_H

#include "Layer.h"

/* Y_i = X
 * just mirror input X into a lot of replicated Y_i
 * dL/dX = \sum dL/dY
 * BranchLayer has store data function, as a data buffer.
 * */

class BranchLayer : public Layer {
public:
    BranchLayer(const int id, const string& name, Layer *prevLayer);
    ~BranchLayer();

    list<Layer*> m_nextLayers;

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void averageParaGradient(const int batchSize);
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);


    virtual  void initializeLRs(const float lr);
    virtual  void updateLRs(const float deltaLoss);
    virtual  void updateParameters(Optimizer* pOptimizer);


    void addNextLayer(Layer* nextLayer);
    virtual  int getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct();
};



#endif //CDLF_FRAME_BRANCHLAYER_H
