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

    void forwardPropagate();
    void backwardPropagate();
    void addLayer(Layer* layer);


private:
    list<Layer*> m_layers;



};


#endif //RL_NONCONVEX_NET_H
