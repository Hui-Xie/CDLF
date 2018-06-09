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

    float getLoss();
    float getAvgLoss();
    float setAvgLoss(const float avgLoss);

    virtual  void initialize(const string& initialMethod);
    virtual  void forward();
    virtual  void backward();
    virtual  void updateParameters(const float lr, const string& method);

    void printGroundTruth();

private:
    float lossCompute();
    void  gradientCompute(const float avgLoss);
    float m_loss;
    float m_avgLoss;

};


#endif //RL_NONCONVEX_LOSSLAYER_H
