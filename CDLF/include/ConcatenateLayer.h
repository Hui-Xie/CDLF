//
// Created by Hui Xie on 12/21/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_CONCATENATELAYER_H
#define CDLF_FRAMEWORK_CONCATENATELAYER_H

#include <Layer.h>

class ConcatenateLayer : public Layer {
public:
    ConcatenateLayer(const int id, const string& name, const vector<Layer*>& pLayersVec);
    ~ConcatenateLayer();

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);
    virtual  void updateParameters(const float lr, const string& method, const int batchSize =1);
    virtual  int getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct(const int layerIndex);

    vector<Layer*> m_pLayersVec;

private:
    vector<int> m_layerLengthVec;
    vector<int> m_layerOffsetVec;

    inline float& dX(const int index) const;

};


#endif //CDLF_FRAMEWORK_CONCATENATELAYER_H