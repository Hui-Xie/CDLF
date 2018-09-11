//
// Created by Hui Xie on 7/28/2018.
//

#ifndef RL_NONCONVEX_SOFTMAXLAYER_H
#define RL_NONCONVEX_SOFTMAXLAYER_H

#include "Layer.h"

/* SoftMax Layer only suit for mutually exclusive output classification in each data point
 * This softmaxLayer support N-d tensor. It implements softmax on the 0th dimension of the N-d tensor.
 * Y_i = exp(X_i)/ (\sum exp(x_j))
 * dL/dx_i = \sum(dL/dy_i * dy_i/dx_i) = exp(x_i)*{dL/dy_i * \sum exp(x_j)-\sum(dL/dy_j*exp(x_j)}/(\sum exp(x_j))^2
 *  where j is the range variable between 1 to N.
 *
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
