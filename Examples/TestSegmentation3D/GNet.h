//
// Created by Hui Xie on 9/18/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_GNET_H
#define CDLF_FRAMEWORK_GNET_H

#include "FeedForwardNet.h"

//Generative Network

class GNet : public FeedForwardNet {
public:
    GNet(const string& name);
    ~GNet();

    virtual void build();
    virtual void train();
    virtual float test();

    Layer* m_pGxLayer;

    string m_name;

};


#endif //CDLF_FRAMEWORK_GNET_H
