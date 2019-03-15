//
// Created by Hui Xie on 6/13/2018.
//

#ifndef CDLF_FRAME_LOSSCONVEXEXAMPLE1_H
#define CDLF_FRAME_LOSSCONVEXEXAMPLE1_H
#include "LossLayer.h"

class LossConvexExample1 : public LossLayer {
public:
    LossConvexExample1(const int id, const string& name,  Layer *prevLayer);
    ~LossConvexExample1();
    virtual void  printGroundTruth();


private:
    virtual float lossCompute();
    virtual void  gradientCompute();


};


#endif //CDLF_FRAME_LOSSCONVEXEXAMPLE1_H
