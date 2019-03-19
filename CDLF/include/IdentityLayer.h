//
// Created by Hui Xie on 8/3/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_IDENTITYLAYER_H
#define CDLF_FRAME_IDENTITYLAYER_H
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
    virtual  void averageParaGradient(const int batchSize);
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);
    virtual  void updateParameters(const float lr, const string& method);
    virtual  int getNumParameters();

    virtual  void initializeLRs(const float lr);
    virtual  void updateLRs(const float deltaLoss);
    virtual  void updateParameters(const string& method);

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct();
};

#endif //CDLF_FRAME_IDENTITYLAYER_H
