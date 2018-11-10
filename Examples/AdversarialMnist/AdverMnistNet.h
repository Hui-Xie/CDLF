//
// Created by Hui Xie on 11/9/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_ADVERMNISTNET_H
#define CDLF_FRAMEWORK_ADVERMNISTNET_H


#include <FeedForwardNet.h>


class AdverMnistNet: public FeedForwardNet {
public:
    AdverMnistNet(const string& name, const string& saveDir);
    ~AdverMnistNet();

    void constructGroundTruth(const int labelValue, Tensor<float>& groundTruth);

    virtual void build();
    virtual void train();
    virtual float test();

    int predict();

    Tensor<float> m_inputTensor;
    Tensor<float> m_groundTruth;

};


#endif //CDLF_FRAMEWORK_ADVERMNISTNET_H
