//
// Created by Hui Xie on 9/15/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_NET_H
#define CDLF_FRAMEWORK_NET_H

#include "Layer.h"
#include <map>
#include <vector>
#include "InputLayer.h"
#include "LossLayer.h"
#include "Optimizer.h"

struct LayerStruct{
    int m_id;
    string m_type;
    string m_name;
    int m_preLayerID;
    vector<int> m_preLayersIDs;
    vector<int> m_outputTensorSize;
    vector<int> m_filterSize;
    vector<int> m_stride;
    int m_numFilter;

    // k in SigmoidLayer and RescaleLayer and ReLU, lambda in SquareLossLayer, initialValue in PaddingLayer
    float m_k;

    // startPosition in SubTensorLayer,  or min_max in ClipLayer
    vector<int> m_startPosition;
};


class Net {
public:
    Net(const string& saveDir);
    ~Net();

    float m_loss;
    float m_batchLoss;
    float m_lastBatchLoss;

    void setLearningRate(const float learningRate);
    void setLossTolerance(const float tolerance);
    void setJudgeLoss(const bool judgeLoss);
    void setBatchSize(const int batchSize);
    void setEpoch(const int epoch);
    void setDir(const string dir);
    void setUnlearningLayerID(const int id);
    void setOneSampleTrain(bool oneSample);

    string getName();

    float getLearningRate();
    float getLossTolerance();
    bool getJudgeLoss();
    int  getBatchSize();
    int getEpoch();
    map<int, Layer*> getLayersMap();
    string getDir();
    int getUnlearningLayerID();
    bool getOneSampleTrain();

    int getNumParameters();


    void addLayer(Layer* layer);
    Layer* getLayer(const int ID);
    vector<Layer*> getLayers(const vector<int> IDVec);


    InputLayer* getInputLayer();
    Layer* getFirstLayer();
    Layer* getFinalLayer();


    void initialize();
    void zeroParaGradient();
    void averageParaGradient(const int batchSize);

    void printIteration(LossLayer* lossLayer,const int nIter, const bool transpose = false);
    void printLayersY();
    void printLayersDY();
    void printArchitecture();

    void saveLayersStruct();
    void loadLayersStruct();
    void saveLayersParameters();
    void loadLayersParameters();
    void saveNetParameters();
    void loadNetParameters();

    //for debug
    void saveYTensor();
    void savedYTensor();

    virtual void save();
    virtual void load();

protected:
    string m_name;
    float m_lossTolerance;
    bool m_judgeLoss;
    int m_batchSize;
    int m_epoch;
    map<int, Layer*> m_layers;
    // check whether layer pointer and name are duplicated
    bool layerExist(const Layer* layer);
    string m_directory;

    // One Sample Train to verify whether network converge
    bool m_OneSampleTrain;

    Optimizer * m_optimizer;

    // layer with layerID < m_unlearningLayerID will not learn;
    // layer with layerID = m_unlearningLayerID will not compute dx of its previous layer
    int m_unlearningLayerID;

    void readLayesStruct(vector<struct LayerStruct>& layersStructVec);
    void createLayers(const vector<struct LayerStruct>& layersStructVec);


};


#endif //CDLF_FRAMEWORK_NET_H
