//
// Created by Sheen156 on 6/5/2018.
//

#ifndef RL_NONCONVEX_NET_H
#define RL_NONCONVEX_NET_H

#include "Layer.h"
#include <list>


class Net {
public:
    Net();
    ~Net();
    int m_batchSize;

    void forwardPropagate();
    void backwardPropagate();
    void addLayer(Layer* layer);


    void sgd(const float lr);



private:
    list<Layer*> m_layers;



};


#endif //RL_NONCONVEX_NET_H
