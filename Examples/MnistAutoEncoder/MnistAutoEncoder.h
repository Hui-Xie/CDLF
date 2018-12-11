//
// Created by Hui Xie on 12/11/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_ADVERMNISTNET_H
#define CDLF_FRAMEWORK_ADVERMNISTNET_H


#include <CDLF.h>
#include "MNIST.h"


class MnistAutoEncoder: public FeedForwardNet {
public:
    MnistAutoEncoder(const string& name, const string& saveDir, MNIST* pMnistData);
    ~MnistAutoEncoder();

    Tensor<float> constructGroundTruth(Tensor<unsigned char> *pLabels, const long index);

    MNIST* m_pMnistData;

    virtual void build();
    virtual void train();
    virtual float test();

};


#endif //CDLF_FRAMEWORK_ADVERMNISTNET_H
