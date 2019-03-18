//
// Created by Hui Xie on 6/5/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef RL_FEEDFORWARDNET_H
#define RL_FEEDFORWARDNET_H

#include "Net.h"
using namespace std;

class FeedForwardNet : public Net {
public:
    FeedForwardNet(const string& saveDir);
    ~FeedForwardNet();

    void forwardPropagate();
    void backwardPropagate(bool computeW);
    void sgd(const float lr, const int batchSize);

    void initializeLRs(const float lr);
    void updateLearingRates(const float deltaLoss, const int batchSize);
    void sgd(const int batchSize); // sgd with various learning rates


    virtual void build() = 0;
    virtual void train() = 0;
    virtual float test() = 0;
};


#endif //RL_FEEDFORWARDNET_H
