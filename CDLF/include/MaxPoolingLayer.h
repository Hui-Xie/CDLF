//
// Created by Hui Xie on 7/28/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_MAXPOOLINGLAYER_H
#define CDLF_FRAME_MAXPOOLINGLAYER_H

#include "Layer.h"

/*
 * Notes: max-pooling can simply be replaced by a convolutional layer with increased stride
 *      without loss in accuracy on several image recognition benchmarks.
 *
 *      current: MaxPoolingLayer does not support dimension collapse.
 *
 * */

class MaxPoolingLayer :  public Layer {
public:
    MaxPoolingLayer(const int id, const string& name, Layer* prevLayer, const vector<int>& filterSize,
                     const vector<int>& stride);
    ~MaxPoolingLayer();

    vector<int> m_filterSize;
    vector<int> m_stride;
    void constructY();


    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void averageParaGradient(const int batchSize);
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);

    virtual  void initializeLRs(const float lr);
    virtual  void updateLRs(const float deltaLoss);
    virtual  void updateParameters(const string& method, Optimizer* pOptimizer);

    virtual  int getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct();

private:
    int m_OneFilterN;


};


#endif //CDLF_FRAME_MAXPOOLINGLAYER_H
