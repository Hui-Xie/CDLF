//
// Created by Hui Xie on 9/18/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_GNET_H
#define CDLF_FRAMEWORK_GNET_H

#include "FeedForwardNet.h"
#include "CrossEntropyLoss.h"

/* Generative Network
 * the derived class of GNet is responsible to assign m_pGxLayer and m_pInputXLayer in its build method
 *
 * */

class GNet  : public FeedForwardNet {
public:
    GNet(const string& name);
    ~GNet();

    InputLayer* m_pInputXLayer;
    Layer* m_pGxLayer;
    CrossEntropyLoss* m_pLossLayer;

};

#endif //CDLF_FRAMEWORK_GNET_H
