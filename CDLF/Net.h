//
// Created by Hui Xie on 6/5/2018.
//

#ifndef RL_NONCONVEX_NET_H
#define RL_NONCONVEX_NET_H

#include "Layer.h"
#include <map>
#include <vector>
#include "LossLayer.h"
using namespace std;

class Net {
public:
    Net();
    ~Net();

    void setLearningRate(const float learningRate);
    void setLossTolerance(const float tolerance);
    void setMaxIteration(const long maxIteration);
    void setJudgeLoss(const bool judgeLoss);
    void setBatchSize(const int batchSize);

    float getLearningRate();
    float getLossTolerance();
    long getMaxIteration();
    bool getJudgeLoss();
    int  getBatchSize();


    void forwardPropagate();
    void backwardPropagate();
    void zeroParaGradient();
    void addLayer(Layer* layer);
    void sgd(const float lr, const int batchSize);

    Layer* getInputLayer();
    Layer* getFinalLayer();

    //Notes: this layerWidthVector does not include LossLayer
    void buildFullConnectedNet(const vector<long> layerWidthVector);
    void initialize();
    void train();
    void printIteration(LossLayer* lossLayer,const int nIter);
    void printLayersY();
    void printLayersDY();
    void printLayersWdW();
    void printArchitecture();


private:
    map<int, Layer*> m_layers;
    float m_learningRate;
    float m_lossTolerance;
    long m_maxIteration;
    bool m_judgeLoss;
    int m_batchSize;
};


#endif //RL_NONCONVEX_NET_H
