//
// Created by Hui Xie on 8/8/2018.
// Copyrigh (c) 2018 Hui Xie. All rights reserved.

#ifndef CDLF_FRAMEWORK_VECTORIZATIONLAYER_H
#define CDLF_FRAMEWORK_VECTORIZATIONLAYER_H

#include "Layer.h"

/*  Y = X.vectorize();
 *  dL/dx = dL/dy
 * */

class VectorizationLayer : public Layer {
public:
    VectorizationLayer(const int id, const string& name,Layer* prevLayer);
    ~VectorizationLayer();

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward();
    virtual  void updateParameters(const float lr, const string& method, const int batchSize =1);
};



#endif //CDLF_FRAMEWORK_VECTORIZATIONLAYER_H
