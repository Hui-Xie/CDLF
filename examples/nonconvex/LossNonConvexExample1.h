//
// Created by Hui Xie on 6/13/2018.
//

#ifndef RL_NONCONVEX_LOSSNONCONVEXEXAMPLE1_H
#define RL_NONCONVEX_LOSSNONCONVEXEXAMPLE1_H
#include "LossLayer.h"

class LossNonConvexExample1: public LossLayer {
public:
    LossNonConvexExample1(const int id, const string& name,  Layer *prevLaye);
    ~LossNonConvexExample1();
    virtual void  printGroundTruth();

private:
    virtual float lossCompute();
    virtual void  gradientCompute();


};


#endif //RL_NONCONVEX_LOSSNONCONVEXEXAMPLE1_H
