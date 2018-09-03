//
// Created by Hui Xie on 6/5/2018.
//

#ifndef RL_NONCONVEX_NET_H
#define RL_NONCONVEX_NET_H

#include "Layer.h"
#include <map>
#include <vector>
#include "LossLayer.h"
#include "InputLayer.h"
using namespace std;

class Net {
public:
    Net();
    ~Net();

    void setLearningRate(const float learningRate);
    void setLossTolerance(const float tolerance);
    void setJudgeLoss(const bool judgeLoss);
    void setBatchSize(const int batchSize);
    void setEpoch(const long epoch);

    float getLearningRate();
    float getLossTolerance();
    bool getJudgeLoss();
    int  getBatchSize();
    long getEpoch();
    map<int, Layer*> getLayersMap();


    void forwardPropagate();
    void backwardPropagate();
    void zeroParaGradient();
    void addLayer(Layer* layer);
    void sgd(const float lr, const int batchSize);

    InputLayer* getInputLayer();
    Layer* getFirstLayer();
    Layer* getFinalLayer();


    void initialize();
    virtual void build() = 0;
    virtual void train() = 0;
    virtual float test() = 0;

    void printIteration(LossLayer* lossLayer,const int nIter);
    void printLayersY();
    void printLayersDY();
    void printLayersWdW();
    void printArchitecture();

protected:
    float m_learningRate;
    float m_lossTolerance;
    bool m_judgeLoss;
    int m_batchSize;
    long m_epoch;

private:
    map<int, Layer*> m_layers;
    bool layerNameExist(const string& name);

};


#endif //RL_NONCONVEX_NET_H
