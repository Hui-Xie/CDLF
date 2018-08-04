//
// Created by Hui Xie on 8/4/2018.
//

#ifndef RL_NONCONVEX_SIGMOIDLAYER_H
#define RL_NONCONVEX_SIGMOIDLAYER_H

#include "Layer.h"

/* Y = exp(x)/( 1+ exp(x)) in element-wise
 * dL/dx = dL/dY * dY/dx = dL/dY * exp(x)/(1 +exp(x))^2
 * */

class SigmoidLayer : public Layer {
public:
    SigmoidLayer(const int id, const string& name,Layer* prevLayer);
    ~SigmoidLayer();

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward();
    virtual  void updateParameters(const float lr, const string& method, const int batchSize =1);
};


#endif //RL_NONCONVEX_SIGMOIDLAYER_H
