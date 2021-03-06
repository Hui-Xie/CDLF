//
// Created by Hui Xie on 8/16/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_CONVEXNET_H
#define CDLF_FRAMEWORK_CONVEXNET_H

#include "CDLF.h"

class ConvexNet : public FeedForwardNet {
public:
    ConvexNet(const string& saveDir);
    ~ConvexNet();

    virtual void build();
    virtual void train();
    virtual float test();


};


#endif //CDLF_FRAMEWORK_CONVEXNET_H
