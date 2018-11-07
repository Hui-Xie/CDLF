//
// Created by Hui Xie on 9/15/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_NET_H
#define CDLF_FRAMEWORK_NET_H

#include "Layer.h"
#include <map>
#include <vector>
#include "InputLayer.h"
#include "LossLayer.h"

class Net {
public:
    Net(const string& name);
    ~Net();

    void setLearningRate(const float learningRate);
    void setLossTolerance(const float tolerance);
    void setJudgeLoss(const bool judgeLoss);
    void setBatchSize(const int batchSize);
    void setEpoch(const long epoch);
    void setDir(const string dir);


    string getName();

    float getLearningRate();
    float getLossTolerance();
    bool getJudgeLoss();
    int  getBatchSize();
    long getEpoch();
    map<int, Layer*> getLayersMap();
    string getDir();

    long getNumParameters();


    void addLayer(Layer* layer);
    Layer* getLayer(const int ID);


    InputLayer* getInputLayer();
    Layer* getFirstLayer();
    Layer* getFinalLayer();


    void initialize();
    void zeroParaGradient();

    void printIteration(LossLayer* lossLayer,const int nIter);
    void printLayersY();
    void printLayersDY();
    void printArchitecture();

    void saveLayersArchitect();
    void loadlayersArchitect();
    void saveLayersParameters();
    void loadLayersParameters();
    void saveNetParameters();
    void loadNetParameters();

    void save();
    void load();

protected:
    string m_name;
    float m_learningRate;
    float m_lossTolerance;
    bool m_judgeLoss;
    int m_batchSize;
    long m_epoch;
    map<int, Layer*> m_layers;
    // check whether layer pointer and name are duplicated
    bool layerExist(const Layer* layer);
    string m_directory;

};


#endif //CDLF_FRAMEWORK_NET_H
