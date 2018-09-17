//
// Created by Hui Xie on 9/15/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_GANET_H
#define CDLF_FRAMEWORK_GANET_H

#include "Net.h"

class GANet : public Net {
public:
    GANet();
    ~GANet();

    virtual void buildG() = 0;
    virtual void buildD() = 0;
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

    bool checkLayerAttribute();

    InputLayer* m_pInputLayer;
    InputLayer* m_pGTLayer; // groundtruth Layer
    Layer* m_pGxLayer; // the output of G, and also input to D.

};


#endif //CDLF_FRAMEWORK_GANET_H
