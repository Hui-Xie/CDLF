//
// Created by Sheen156 on 6/5/2018.
//

#ifndef RL_NONCONVEX_NET_H
#define RL_NONCONVEX_NET_H

#include "Layer.h"
#include <list>
#include <vector>
#include "LossLayer.h"
using namespace std;

class Net {
public:
    Net();
    ~Net();

    void setBatchSize(const int batchSize);
    void setLearningRate(const float learningRate);
    void setLossTolerance(const float tolerance);
    void setMaxItration(const int maxIteration);

    void forwardPropagate();
    void backwardPropagate();
    void addLayer(Layer* layer);
    void sgd(const float lr);

    //Notes: this layerWidthVector does not include LossLayer
    void buildNet(const vector<long> layerWidthVector);
    void initialize();
    void train(const int nIteration);
    void printIteration(LossLayer* lossLayer,const int nIter);
    void printLayersY();
    void printLayersDY();


private:
    list<Layer*> m_layers;
    int m_batchSize;
    float m_learningRate;
    float m_lossTolerance;
    float m_maxIteration;
};


#endif //RL_NONCONVEX_NET_H
