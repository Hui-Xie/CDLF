//
// Created by Sheen156 on 6/6/2018.
//

#ifndef RL_NONCONVEX_LOSSLAYER_H
#define RL_NONCONVEX_LOSSLAYER_H

#include "Layer.h"

class LossLayer : public Layer {
public:
    LossLayer(Layer* preLayer);
    ~LossLayer();
    float m_loss;

    virtual  void forward();
    virtual  void backward();
    virtual  void initialize(const string& initialMethod);

private:
    float lossCompute();
    void  gradientCompute();

};


#endif //RL_NONCONVEX_LOSSLAYER_H
