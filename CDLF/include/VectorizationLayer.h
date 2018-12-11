//
// Created by Hui Xie on 8/8/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef CDLF_FRAMEWORK_VECTORIZATIONLAYER_H
#define CDLF_FRAMEWORK_VECTORIZATIONLAYER_H

#include "ReshapeLayer.h"

/*  Y = X.vectorize();
 *  dL/dx = dL/dy
 * */

class VectorizationLayer : public ReshapeLayer {
public:
    VectorizationLayer(const int id, const string& name,Layer* prevLayer);
    ~VectorizationLayer();
};



#endif //CDLF_FRAMEWORK_VECTORIZATIONLAYER_H
