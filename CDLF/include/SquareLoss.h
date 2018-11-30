//
// Created by Hui Xie on 11/29/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_SQUARELOSS_H
#define CDLF_FRAMEWORK_SQUARELOSS_H

#include "LossLayer.h"

/* L= 0.5*\sum (x_i- g_i)^2
 * where g_i is the groundtruth distribution
 *       x_i is the output of previous layer, e.g. softmax;
 *
 * */

class SquareLoss  : public LossLayer {
public:
    SquareLoss(const int id, const string& name,  Layer *prevLayer);
    ~SquareLoss();
    virtual void  printGroundTruth();


private:
    virtual float lossCompute();
    virtual void  gradientCompute();


};


#endif //CDLF_FRAMEWORK_SQUARELOSS_H
