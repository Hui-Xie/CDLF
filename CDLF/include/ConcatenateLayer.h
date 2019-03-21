//
// Created by Hui Xie on 12/21/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_CONCATENATELAYER_H
#define CDLF_FRAMEWORK_CONCATENATELAYER_H

#include <Layer.h>

class ConcatenateLayer : public Layer {
public:
    ConcatenateLayer(const int id, const string& name, const vector<Layer*>& layersVec, const vector<int>& tensorSize);
    ~ConcatenateLayer();

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void averageParaGradient(const int batchSize);
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);

    virtual  int  getNumParameters();

    virtual  void initializeLRs(const float lr);
    virtual  void updateLRs(const float deltaLoss);
    virtual  void updateParameters(Optimizer* pOptimizer);

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct();

    vector<Layer*> m_layersVec;

private:
    vector<int> m_layerLengthVec;
    vector<int> m_layerOffsetVec;

    inline float& dX(const int index) const;

};


#endif //CDLF_FRAMEWORK_CONCATENATELAYER_H
