//
// Created by Hui Xie on 9/15/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_GANET_H
#define CDLF_FRAMEWORK_GANET_H

#include "GNet.h"
#include "DNet.h"


class GAN {
public:
    GAN(const string& name, GNet* pGNet, DNet* pDNet);
    ~GAN();

    virtual void trainG(const int N) = 0;
    virtual void trainD(const int N) = 0;
    virtual float testG() = 0;

    void forwardG();
    void forwardD();
    void backwardG();
    void backwardD();

    void switchDToGT();
    void switchDToGx();
    void switchDToStub();
    void setStubLayer(Layer* pStubLayer);

    void copyGxYFromGtoD();
    void copyGxGradientFromDtoG();

    GNet* m_pGNet;
    DNet* m_pDNet;
    Layer* m_pStubLayer;

    string m_name;

};


#endif //CDLF_FRAMEWORK_GANET_H
