//
// Created by Hui Xie on 7/24/2018.
//

#ifndef CDLF_FRAME_LOSSCONVEXEXAMPLE2_H
#define CDLF_FRAME_LOSSCONVEXEXAMPLE2_H

#include "LossLayer.h"

// L(x) = \sum exp(x_i -i)

class LossConvexExample2: public LossLayer {
public:
    LossConvexExample2(const int id, const string& name,Layer *prevLayer);
    ~LossConvexExample2();
    virtual void  printGroundTruth();


private:
    virtual float lossCompute();
    virtual void  gradientCompute();


};

#endif //CDLF_FRAME_LOSSCONVEXEXAMPLE2_H
