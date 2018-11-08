//
// Created by Hui Xie on 6/5/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_FEEDFORWARDNET_H
#define RL_FEEDFORWARDNET_H

#include "Net.h"
using namespace std;

class FeedForwardNet : public Net {
public:
    FeedForwardNet(const string& name, const string& saveDir);
    ~FeedForwardNet();

    void forwardPropagate();
    void backwardPropagate(bool computeW);
    void sgd(const float lr, const int batchSize);

    virtual void build() = 0;
    virtual void train() = 0;
    virtual float test() = 0;
};


#endif //RL_FEEDFORWARDNET_H
