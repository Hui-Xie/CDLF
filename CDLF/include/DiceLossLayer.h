//
// Created by Hui Xie on 01/22/19.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_DICELOSS_H
#define CDLF_FRAMEWORK_DICELOSS_H

#include "LossLayer.h"

/*
 *
 * For preLayer is Sigmoid
 * Loss = 1- Dice
 *      = 1 - 2 |x.* g|/(|x| + |g|)
 * where x is the output vector of previous layer
 *       g is the corresponding binary groundtruth vector containing only 0 or 1 elements
 *       bars indicate L1 norm
 *
 * Notes:
 * 1  Dice computation formula is not a exact match with its set-version formula;
 * 2  the above computing formula is an approximation of set-version formula;
 * 3  In x_i = {0,1} binary case, L1 norm is better than L2 norm in computing dice coefficient;
 * 4  In x_i= (0,1) real number case,  Dice computed by above dice computation formula may be greater or lesser than the real set-version dice;
 *
 * */

class DiceLossLayer  : public LossLayer {
public:
    DiceLossLayer(const int id, const string& name,  Layer *prevLayer);
    ~DiceLossLayer();

private:
    virtual float lossCompute();
    virtual void  gradientCompute();

};
#endif //CDLF_FRAMEWORK_DICELOSS_H
