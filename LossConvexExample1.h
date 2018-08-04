//
// Created by Hui Xie on 6/13/2018.
//

#ifndef RL_NONCONVEX_LOSSCONVEXEXAMPLE1_H
#define RL_NONCONVEX_LOSSCONVEXEXAMPLE1_H
#include "LossLayer.h"

class LossConvexExample1 : public LossLayer {
public:
    LossConvexExample1(const int id, const string& name);
    ~LossConvexExample1();

private:
    virtual float lossCompute();
    virtual void  gradientCompute();
    virtual void  printGroundTruth();

};


#endif //RL_NONCONVEX_LOSSCONVEXEXAMPLE1_H
