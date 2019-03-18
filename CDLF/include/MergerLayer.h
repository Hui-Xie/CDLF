//
// Created by Hui Xie on 8/4/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_SUMLAYER_H
#define CDLF_FRAME_SUMLAYER_H

#include "Layer.h"

/* Y = \sum X_i
 * dL/dx_i = dL/dY * dY/dX_i = dL/dY;
 * in the BranchLayer, the dL/dY should be accumulate from the following layers of MergerLayer
 * MergerLayer can be a switch to on/off connecting to different incoming layer.
 * */

class MergerLayer : public Layer {
public:
    MergerLayer(const int id, const string& name, const vector<int>& tensorSize);
    ~MergerLayer();

    list<Layer*> m_prevLayers;

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void averageParaGradient(const int batchSize);
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);
    virtual  void updateParameters(const float lr, const string& method, const int batchSize=1);

    virtual  void initializeLRs(const float lr);
    virtual  void updateLRs(const float deltaLoss, const int batchSize = 1);
    virtual  void updateParameters(const string& method, const int batchSize=1);

    virtual void addPreviousLayer(Layer* prevLayer);
    void delPreviousLayer(Layer* prevLayer);
    virtual  int getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct();

private:
    bool isLayerInList(const Layer* layer);
    void delLayerFromList(const Layer* layer);
};


#endif //CDLF_FRAME_SUMLAYER_H
