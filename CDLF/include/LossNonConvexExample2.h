//
// Created by Hui Xie on 6/15/2018.
//

#ifndef CDLF_FRAME_LOSSNONCONVEXEXAMPLE2_H
#define CDLF_FRAME_LOSSNONCONVEXEXAMPLE2_H

#include "LossLayer.h"

class LossNonConvexExample2 : public LossLayer {
public:
    LossNonConvexExample2(const int id, const string& name,Layer *prevLayer);
    ~LossNonConvexExample2();
    virtual void  printGroundTruth();

private:
    virtual float lossCompute();
    virtual void  gradientCompute();

};


#endif //CDLF_FRAME_LOSSNONCONVEXEXAMPLE2_H
