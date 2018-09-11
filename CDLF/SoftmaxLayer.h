//
// Created by Hui Xie on 7/28/2018.
//

#ifndef RL_NONCONVEX_SOFTMAXLAYER_H
#define RL_NONCONVEX_SOFTMAXLAYER_H

#include "Layer.h"

/* SoftMax Layer only suit for mutually exclusive output classification in each data point
 * This softmaxLayer support N-d tensor. It implements the softmax on the 0th dimension.
 *
 * */

class SoftmaxLayer : public Layer {
public:
    SoftmaxLayer(const int id, const string& name,Layer* prevLayer);
    ~SoftmaxLayer();

    float m_sumExpX;

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward();
    virtual  void updateParameters(const float lr, const string& method, const int batchSize=1);

};

#endif //RL_NONCONVEX_SOFTMAXLAYER_H
