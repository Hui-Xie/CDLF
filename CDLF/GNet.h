//
// Created by Hui Xie on 9/18/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_GNET_H
#define CDLF_FRAMEWORK_GNET_H

#include "FeedForwardNet.h"

class GNet  : public FeedForwardNet {
public:
    GNet(const string& name);
    ~GNet();

    Layer* m_pGxLayer;
    InputLayer* m_pInputXLayer;

};

#endif //CDLF_FRAMEWORK_GNET_H
