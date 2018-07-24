//
// Created by Sheen156 on 7/24/2018.
//

#ifndef RL_NONCONVEX_LOSSCONVEXEXAMPLE2_H
#define RL_NONCONVEX_LOSSCONVEXEXAMPLE2_H

#include "LossLayer.h"

// L(x) = \sum exp(x_i -i)

class LossConvexExample2: public LossLayer {
public:
    LossConvexExample2(const int id, const string& name);
    ~LossConvexExample2();

private:
    virtual float lossCompute();
    virtual void  gradientCompute();
    virtual void  printGroundTruth();

};

#endif //RL_NONCONVEX_LOSSCONVEXEXAMPLE2_H
