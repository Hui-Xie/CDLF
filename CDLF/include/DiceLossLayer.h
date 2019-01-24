//
// Created by Hui Xie on 01/22/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_DICELOSS_H
#define CDLF_FRAMEWORK_DICELOSS_H

#include "LossLayer.h"

/* Loss = 1- Dice
 *      = 1 - 2 |x.* g|/(|x| + |g|)
 * where x is the output vector of previous layer
 *       g is the corresponding binary groundtruth vector containing only 0 or 1 elements
 *       prim indicate transpose vector;
 *
 * further deduction:
 *  Loss = 1 - 2 sqrt((x .* g)'*(x .*g))/ (sqrt(x'*x)+ sqrt(g'*g))
 *
 * */

class DiceLossLayer  : public LossLayer {
public:
    DiceLossLayer(const int id, const string& name,  Layer *prevLayer);
    ~DiceLossLayer();

private:
    virtual float lossCompute();
    virtual void  gradientCompute();
    float m_xDotg_norm;
    float m_xPlusg_norm;


};
#endif //CDLF_FRAMEWORK_DICELOSS_H
