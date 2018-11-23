//
// Created by Hui Xie on 8/3/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_IDENTITYLAYER_H
#define RL_NONCONVEX_IDENTITYLAYER_H
#include "Layer.h"

/* Identity Layer same with  a residual edge
 * Identity's ID should be between the IDs of its previous layer and its next layer
 * there is no case that there are sum before the Identity Layer.
 * */

class IdentityLayer : public Layer {
public:
    IdentityLayer(const int id, const string& name,Layer* prevLayer);
    ~IdentityLayer();

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward(bool computeW);
    virtual  void updateParameters(const float lr, const string& method, const int batchSize =1);
    virtual  long getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct(const int layerIndex);
};

#endif //RL_NONCONVEX_IDENTITYLAYER_H
