//
// Created by Sheen156 on 6/6/2018.
//

#ifndef RL_NONCONVEX_LOSSLAYER_H
#define RL_NONCONVEX_LOSSLAYER_H

#include "Layer.h"

class LossLayer : public Layer {
public:
    LossLayer(const int id, const string name,Layer* preLayer);
    ~LossLayer();

    float getLoss();

    virtual  void initialize(const string& initialMethod);
    virtual  void forward();
    virtual  void backward();
    virtual  void updateParameters(const float lr, const string& method);

    virtual float lossCompute()=0;
    virtual void  gradientCompute()=0;
    virtual void printGroundTruth()=0;
    float m_loss;

};


#endif //RL_NONCONVEX_LOSSLAYER_H
