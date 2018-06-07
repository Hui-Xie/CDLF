//
// Created by Sheen156 on 6/5/2018.
//

#ifndef RL_NONCONVEX_FCLAYER_H
#define RL_NONCONVEX_FCLAYER_H

#include "Layer.h"
#include  <blaze/Math.h>
using namespace blaze;

// y = W*x+ b
// where y and b is m-D vector, y is output vector;
//       x is n-D input vector
//       W is m*n dimensional matrix
class FCLayer :  public Layer{
public:
    FCLayer(long width, Layer* preLayer);
    ~FCLayer();

    long m_n; //input width
    long m_m; //output width
    DynamicVector<float>*  m_pYVector;
    DynamicMatrix<float>*  m_pW;

    virtual  void forward();
    virtual  void backward();
    virtual  void initialize(const string& initialMethod);



};


#endif //RL_NONCONVEX_FCLAYER_H
