//
// Created by Hui Xie on 6/12/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_NORMALIZATIONLAYER_H
#define CDLF_FRAME_NORMALIZATIONLAYER_H
#include "Layer.h"

// NormalizationLayer only has one previous layer.

class NormalizationLayer: public Layer {
public:
    NormalizationLayer(const int id, const string& name,Layer* prevLayer, const vector<int>& tensorSize);
    ~NormalizationLayer();
    float m_epsilon;
    float* m_pSigma; // standard deviation

    bool m_existFeautureDim;

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void averageParaGradient(const int batchSize);
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);


    virtual  void initializeLRs(const float lr);
    virtual  void updateLRs(const float deltaLoss);
    virtual  void updateParameters(Optimizer* pOptimizer);

    virtual  int getNumParameters();
    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct();

    void setFeatureDim();
    void allocateSigmas();
    void freeSigmas();

};


#endif //CDLF_FRAME_NORMALIZATIONLAYER_H
