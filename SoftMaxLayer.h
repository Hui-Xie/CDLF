//
// Created by Hui Xie on 7/28/2018.
//

#ifndef RL_NONCONVEX_SOFTMAXLAYER_H
#define RL_NONCONVEX_SOFTMAXLAYER_H

#include "Layer.h"

/* SoftMax Layer only suit for mutually exclusive output classificaiton
 * For 3D segmentation output, we need use GroupSoftMax
 *
 * */

class SoftMaxLayer : public Layer {
public:
    SoftMaxLayer(const int id, const string& name,Layer* prevLayer);
    ~SoftMaxLayer();

    float m_sumExpX;

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward();
    virtual  void updateParameters(const float lr, const string& method, const int batchSize=1);

};

#endif //RL_NONCONVEX_SOFTMAXLAYER_H
