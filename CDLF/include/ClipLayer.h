//
// Created by Hui Xie on 03/15/2019.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_CLIPLAYER_H
#define CDLF_FRAME_CLIPLAYER_H

#include "Layer.h"

/*
 * Clip[x,{min,max}]
 * gives x for min≤x≤max, min for x<min and max for x>max.
 *
 * */
class ClipLayer :  public Layer{
public:
    ClipLayer(const int id, const string& name, Layer* prevLayer, const int min, const int max);
    ~ClipLayer();

    int m_min;
    int m_max;

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void averageParaGradient(const int batchSize);
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);
    virtual  void updateParameters(const float lr, const string& method);

    virtual  void initializeLRs(const float lr);
    virtual  void updateLRs(const float deltaLoss);
    virtual  void updateParameters(const string& method);

    virtual  int getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct();
};


#endif //CDLF_FRAME_CLIPLAYER_H
