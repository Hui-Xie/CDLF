//
// Created by Hui Xie on 8/17/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_CONV4DNET_H
#define CDLF_FRAMEWORK_CONV4DNET_H

#include "CDLF.h"

class ConvCudaNet  : public FeedForwardNet {
public:
    ConvCudaNet(const string& saveDir);
    ~ConvCudaNet();

    virtual void build();
    virtual void train();
    virtual float test();

};


#endif //CDLF_FRAMEWORK_CONV4DNET_H
