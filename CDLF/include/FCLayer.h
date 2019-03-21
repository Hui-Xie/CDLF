//
// Created by Hui Xie on 6/5/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_FCLAYER_H
#define CDLF_FRAME_FCLAYER_H

#include "Layer.h"
#include <map>


// y = W*x+ b
// where y and b is m-D vector, y is output vector;
//       x is n-D input vector
//       W is m*n dimensional matrix
class FCLayer :  public Layer{
public:
     FCLayer(const int id, const string& name, Layer* prevLayer, const int outputWidth);
    ~FCLayer();

    int m_m; //output width
    int m_n; //input width
    Tensor<float>*  m_pW;
    Tensor<float>*  m_pB;
    Tensor<float>*  m_pdW;
    Tensor<float>*  m_pdB;

    //Learning Rates
    //Tensor<float>*  m_pWLr;
    //Tensor<float>*  m_pBLr;

    // Adam
    Tensor<float>*  m_pWM;  //1st moment
    Tensor<float>*  m_pBM;
    Tensor<float>*  m_pWR;
    Tensor<float>*  m_pBR; //2nd moment

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void averageParaGradient(const int batchSize);
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);

    virtual void allocateOptimizerMem(const string method);
    virtual void freeOptimizerMem();

    //virtual  void initializeLRs(const float lr);
    //virtual  void updateLRs(const float deltaLoss);
    virtual  void updateParameters(Optimizer* pOptimizer);

    void printWandBVector();
    void printdWanddBVector();
    virtual  int getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct();
};


#endif //CDLF_FRAME_FCLAYER_H
