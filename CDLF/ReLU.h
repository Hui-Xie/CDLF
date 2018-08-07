//
// Created by Hui Xie on 6/7/2018.
//

#ifndef RL_NONCONVEX_RELU_H
#define RL_NONCONVEX_RELU_H
#include "Layer.h"

class ReLU : public Layer {
public:
    ReLU(const int id, const string& name,Layer* prevLayer);
    ~ReLU();

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward();
    virtual  void updateParameters(const float lr, const string& method, const int batchSize =1);
};


#endif //RL_NONCONVEX_RELU_H