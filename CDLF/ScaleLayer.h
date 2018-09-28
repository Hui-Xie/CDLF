//
// Created by Hui Xie on 9/28/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_SCALELAYER_H
#define CDLF_FRAMEWORK_SCALELAYER_H


/* y_i = k*x_i, where k is a learning scalar.
 * dL/dx_i = dL/dy_i *k
 * dL/dk = (dL/dy)' * x, where y and x are 1D vector form,prime symbol means transpose.
 *
 * */

#include "Layer.h"

class ScaleLayer : public Layer {
public:
    ScaleLayer(const int id, const string& name, Layer* prevLayer, const float k=1);
    ~ScaleLayer();

    float m_k;
    float m_dk;

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward(bool computeW);
    virtual  void updateParameters(const float lr, const string& method, const int batchSize=1);
    virtual  long getNumParameters();


};


#endif //CDLF_FRAMEWORK_SCALELAYER_H
