//
// Created by Hui Xie on 7/19/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_BIASLAYER_H
#define CDLF_FRAME_BIASLAYER_H
#include "Layer.h"

/*  Element-wise Linear Layer
 *  It replaces old-version ScaleLayer and LinearLayer.
 *  Y_i = K_i*X_i + B_i  for each element
 *  where Y is the output of Linear Layer
 *        X is the input of the Linear Layer
 *        K is the learning parameter of Scale for each element
 *        B is the learning parameter of Bias for each element
 *  dL/dX = dL/dY * K_i    Where L is Loss
 *  dL/dk = dL/dY * X_i
 *  dL/db = dL/dY
 * */

class LinearLayer : public Layer {
public:
    LinearLayer(const int id, const string& name, Layer* prevLayer);
    ~LinearLayer();

    Tensor<float>*  m_pK;
    Tensor<float>*  m_pB;
    Tensor<float>*  m_pdK;
    Tensor<float>*  m_pdB;

    Tensor<float>*  m_pKM;
    Tensor<float>*  m_pKR;
    Tensor<float>*  m_pBM;
    Tensor<float>*  m_pBR;

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void averageParaGradient(const int batchSize);
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);


    //virtual  void initializeLRs(const float lr);
    //virtual  void updateLRs(const float deltaLoss);
    virtual void allocateOptimizerMem(const string method);
    virtual void freeOptimizerMem();
    virtual  void updateParameters(const string& method, Optimizer* pOptimizer);

    virtual  int getNumParameters();
    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct();
};


#endif //CDLF_FRAME_BIASLAYER_H
