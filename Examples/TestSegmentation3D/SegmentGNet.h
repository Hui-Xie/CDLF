//
// Created by Hui Xie on 9/18/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_SEGMENTGNET_H
#define CDLF_FRAMEWORK_SEGMENTGNET_H

#include "GNet.h"


/* Generative Network
 * the derived class of GNet is responsible to assign m_pGxLayer and m_pInputXLayer in its build method
 *
 * */


class SegmentGNet : public GNet {
public:
    SegmentGNet(const string& saveDir);
    ~SegmentGNet();

    virtual void build();
    virtual void train();
    virtual float test();
};


#endif //CDLF_FRAMEWORK_SEGMENTGNET_H
