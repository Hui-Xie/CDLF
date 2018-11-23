//
// Created by Hui Xie on 7/28/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_MAXPOOLINGLAYER_H
#define RL_NONCONVEX_MAXPOOLINGLAYER_H

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
    MaxPoolingLayer(const int id, const string& name, Layer* prevLayer, const vector<long>& filterSize,
                     const int stride=1);
    ~MaxPoolingLayer();

    vector<long> m_filterSize;
    void constructY();


    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);
    virtual  void updateParameters(const float lr, const string& method, const int batchSize =1);
    virtual  long getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct(const int layerIndex);

private:
    int m_stride;
    int m_OneFilterN;


};


#endif //RL_NONCONVEX_MAXPOOLINGLAYER_H
