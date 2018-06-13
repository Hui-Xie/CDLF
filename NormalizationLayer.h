//
// Created by Sheen156 on 6/12/2018.
//

#ifndef RL_NONCONVEX_NORMALIZATIONLAYER_H
#define RL_NONCONVEX_NORMALIZATIONLAYER_H
#include "Layer.h"

class NormalizationLayer: public Layer {
    NormalizationLayer(Layer* preLayer);
    ~NormalizationLayer();

    virtual  void initialize(const string& initialMethod);
    virtual  void forward();
    virtual  void backward();
    virtual  void updateParameters(const float lr, const string& method);

};


#endif //RL_NONCONVEX_NORMALIZATIONLAYER_H
