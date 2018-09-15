//
// Created by Hui Xie on 8/16/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_CONVEXNET_H
#define CDLF_FRAMEWORK_CONVEXNET_H

#include "CDLF.h"

class NonconvexNet : public FeedForwardNet {
public:
    NonconvexNet(const vector<long>& layerWidthVector);
    ~NonconvexNet();

    virtual void build();
    virtual void train();
    virtual float test();

    vector<long> m_layerWidthVector;
};


#endif //CDLF_FRAMEWORK_CONVEXNET_H
