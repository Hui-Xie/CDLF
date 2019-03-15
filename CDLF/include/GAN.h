//
// Created by Hui Xie on 9/15/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_GANET_H
#define CDLF_FRAMEWORK_GANET_H

#include "GNet.h"
#include "DNet.h"


class GAN {
public:
    GAN(const string& name, GNet* pGNet, DNet* pDNet);
    ~GAN();

    virtual void quicklySwitchTrainG_D() = 0;
    virtual void trainG() = 0;
    virtual void trainD() = 0;
    virtual float testG(bool outputFile) = 0;

    string getName();

    void forwardG();
    void forwardD();
    void backwardG();
    void backwardD();

    void switchDtoGT();
    void switchDtoGx();
    void switchDtoStub();
    void setStubLayer(Layer* pStubLayer);

    void copyGxYFromGtoD();
    void copyGxGradientFromDtoG();

    GNet* m_pGNet;
    DNet* m_pDNet;
    Layer* m_pStubLayer;

    string m_name;

};


#endif //CDLF_FRAMEWORK_GANET_H
