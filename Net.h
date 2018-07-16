//
// Created by Sheen156 on 6/5/2018.
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
    void setMaxIteration(const int maxIteration);
    void setJudgeLoss(const bool judgeLoss);

    void forwardPropagate();
    void backwardPropagate();
    void addLayer(Layer* layer);
    void sgd(const float lr);

    //Notes: this layerWidthVector does not include LossLayer
    void buildNet(const vector<long> layerWidthVector, Layer* lossLayer);
    void initialize();
    void train();
    void printIteration(LossLayer* lossLayer,const int nIter);
    void printLayersY();
    void printLayersDY();
    void printLayersWdW();


private:
    map<int, Layer*> m_layers;
    float m_learningRate;
    float m_lossTolerance;
    float m_maxIteration;
    bool m_judgeLoss;
};


#endif //RL_NONCONVEX_NET_H
