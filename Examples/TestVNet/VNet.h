//
// Created by Hui Xie on 8/16/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_CONVNET_H
#define CDLF_FRAMEWORK_CONVNET_H

#include "CDLF.h"

class VNet : public FeedForwardNet {
public:
    VNet(const string& saveDir);
    ~VNet();

    virtual void build();
    virtual void train();
    virtual float test();

};


#endif //CDLF_FRAMEWORK_CONVNET_H
