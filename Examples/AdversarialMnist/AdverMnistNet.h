//
// Created by Hui Xie on 11/9/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_ADVERMNISTNET_H
#define CDLF_FRAMEWORK_ADVERMNISTNET_H


#include <FeedForwardNet.h>


class AdverMnistNet: public FeedForwardNet {
public:
    AdverMnistNet(const string& saveDir);
    ~AdverMnistNet();

    void constructGroundTruth(const int labelValue, Tensor<float>& groundTruth);

    virtual void build();
    virtual void train();
    virtual float test();

    int predict(const Tensor<float>& inputTensor);

    Tensor<float> m_adversaryTensor;
    Tensor<float> m_groundTruth;
    Tensor<float> m_originTensor;

    // regulization coefficient Loss = Loss(f(x)) + 0.5*lambda*(x-x0)^2
    // it addes backpropagation gardient lambda*(x-x0)
    float m_lambda;

    void setLambda(float lambda);

    void trimAdversaryTensor();
    void saveInputDY(const string filename);

};


#endif //CDLF_FRAMEWORK_ADVERMNISTNET_H
