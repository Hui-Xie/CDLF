//
// Created by Sheen156 on 6/5/2018.
//

#ifndef RL_NONCONVEX_FCLAYER_H
#define RL_NONCONVEX_FCLAYER_H

#include "Layer.h"
#include  "d:\\blaze-3.3\\blaze\\Math.h"
using blaze::DynamicVector;

// y = W*x+ b
// where y and b is m-D vector, y is output vector;
//       x is n-D input vector
//       W is m*n dimensional matrix
class FCLayer :public Layer{
public:
    FCLayer(long width, Layer* preLayer);
    ~FCLayer();

    long m_n; //input width
    long m_m; //output width


};


#endif //RL_NONCONVEX_FCLAYER_H
