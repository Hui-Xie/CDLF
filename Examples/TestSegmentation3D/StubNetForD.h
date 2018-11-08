//
// Created by Hui Xie on 9/21/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_STUBFORDNET_H
#define CDLF_FRAMEWORK_STUBFORDNET_H

#include "FeedForwardNet.h"

class StubNetForD: public FeedForwardNet {
public:
    StubNetForD(const string& name, const string& saveDir);
    ~StubNetForD();

    virtual void build();
    virtual void train();
    virtual float test();

    void randomOutput();
    Tensor<float>* getOutput();
};


#endif //CDLF_FRAMEWORK_STUBFORDNET_H
