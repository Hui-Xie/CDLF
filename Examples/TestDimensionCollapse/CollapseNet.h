//
// Created by Hui Xie on 8/17/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_COLLAPSENET_H
#define CDLF_FRAMEWORK_COLLAPSENET_H

#include "CDLF.h"

class CollapseNet  : public FeedForwardNet {
public:
    CollapseNet();
    ~CollapseNet();

    virtual void build();
    virtual void train();
    virtual float test();

};


#endif //CDLF_FRAMEWORK_COLLAPSENET_H
