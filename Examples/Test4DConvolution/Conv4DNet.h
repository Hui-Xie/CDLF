//
// Created by Hui Xie on 8/17/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_CONV4DNET_H
#define CDLF_FRAMEWORK_CONV4DNET_H

#include "FeedForwardNet.h"

class Conv4DNet  : public FeedForwardNet {
public:
    Conv4DNet(const string& name, const string& saveDir);
    ~Conv4DNet();

    virtual void build();
    virtual void train();
    virtual float test();

};


#endif //CDLF_FRAMEWORK_CONV4DNET_H
