//
// Created by Hui Xie on 9/15/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_GANET_H
#define CDLF_FRAMEWORK_GANET_H

#include "FeedForwardNet.h"

class GANet : public FeedForwardNet {
public:
    GANet();
    ~GANet();

    virtual void buildG() = 0;
    virtual void buildD() = 0;
    virtual void trainG() = 0;
    virtual void trainD() = 0;
    virtual float test() = 0;

    virtual void build();
    virtual void train();

};


#endif //CDLF_FRAMEWORK_GANET_H
