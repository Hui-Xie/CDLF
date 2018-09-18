//
// Created by Hui Xie on 9/18/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_SEGMENTGNET_H
#define CDLF_FRAMEWORK_SEGMENTGNET_H

#include "GNet.h"

//Generative Network

class SegmentGNet : public GNet {
public:
    SegmentGNet(const string& name);
    ~SegmentGNet();

    virtual void build();
    virtual void train();
    virtual float test();
};


#endif //CDLF_FRAMEWORK_SEGMENTGNET_H
