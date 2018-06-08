//
// Created by Sheen156 on 6/7/2018.
//

#ifndef RL_NONCONVEX_RELU_H
#define RL_NONCONVEX_RELU_H
#include "Layer.h"

class ReLU : public Layer {
public:
    ReLU(Layer* preLayer);
    ~ReLU();

    virtual  void forward();
    virtual  void backward();
    virtual  void initialize(const string& initialMethod);
};


#endif //RL_NONCONVEX_RELU_H
