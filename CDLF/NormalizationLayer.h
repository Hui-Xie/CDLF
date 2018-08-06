//
// Created by Hui Xie on 6/12/2018.
//

#ifndef RL_NONCONVEX_NORMALIZATIONLAYER_H
#define RL_NONCONVEX_NORMALIZATIONLAYER_H
#include "Layer.h"

// NormalizationLayer only has one previous layer.

class NormalizationLayer: public Layer {
public:
    NormalizationLayer(const int id, const string& name,Layer* prevLayer);
    ~NormalizationLayer();
    float m_epsilon;

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward();
    virtual  void updateParameters(const float lr, const string& method, const int batchSize =1);

};


#endif //RL_NONCONVEX_NORMALIZATIONLAYER_H
