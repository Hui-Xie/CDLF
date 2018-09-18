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

    virtual void trainG() = 0;
    virtual void trainD() = 0;
    virtual float test() = 0;

    void forwardG();
    void forwardD();
    void backwardG();
    void backwardD();
    void sgdG();
    void sgdD();

    void switchToGT();
    void switchToGx();

    void copyGxYFromGtoD();
    void copyGxGradientFromDtoG();

    GNet* m_pGNet;
    DNet* m_pDNet;

    string m_name;

};


#endif //CDLF_FRAMEWORK_GANET_H
