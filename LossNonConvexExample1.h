//
// Created by Sheen156 on 6/13/2018.
//

#ifndef RL_NONCONVEX_LOSSNONCONVEXEXAMPLE1_H
#define RL_NONCONVEX_LOSSNONCONVEXEXAMPLE1_H
#include "LossLayer.h"

class LossNonConvexExample1 {
public:
    LossNonConvexExample1(Layer* preLayer);
    ~LossNonConvexExample1();

private:
    virtual float lossCompute();
    virtual void  gradientCompute();
    virtual void  printGroundTruth();

};


#endif //RL_NONCONVEX_LOSSNONCONVEXEXAMPLE1_H
