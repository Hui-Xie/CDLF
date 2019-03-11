//
// Created by Hui Xie on 01/15/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_DECONVNET_H
#define CDLF_DECONVNET_H

#include "CDLF.h"


class DeConvNet : public FeedForwardNet {
public:
    DeConvNet(const string& saveDir);
    ~DeConvNet();

    virtual void build();
    virtual void train();
    virtual float test();

};


#endif //CDLF_DECONVNET_H