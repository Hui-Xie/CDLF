//
// Created by Hui Xie on 9/18/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_DNET_H
#define CDLF_FRAMEWORK_DNET_H

#include "FeedForwardNet.h"

//Discriminate Network

class DNet : public FeedForwardNet {
public:
    DNet(const string& name);
    ~DNet();

    virtual void build();
    virtual void train();
    virtual float test();

    Layer* m_pGTLayer;
    Layer* m_pGxLayer;
    Layer* m_pInputXLayer;


};


#endif //CDLF_FRAMEWORK_DNET_H
