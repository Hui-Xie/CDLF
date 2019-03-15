//
// Created by Hui Xie on 12/11/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_ADVERMNISTNET_H
#define CDLF_FRAMEWORK_ADVERMNISTNET_H


#include <CDLF.h>
#include "MNIST.h"


class MnistAutoEncoder: public FeedForwardNet {
public:
    MnistAutoEncoder(const string& saveDir, MNIST* pMnistData);
    ~MnistAutoEncoder();

    MNIST* m_pMnistData;

    virtual void build();
    virtual void train();
    virtual float test();

    void autoEncode(const Tensor<float>& inputImage, int& predictLabel, Tensor<float>& reconstructImage);

    void outputLayer(const Tensor<float>& inputImage, const int layID, const string filename);

};


#endif //CDLF_FRAMEWORK_ADVERMNISTNET_H
