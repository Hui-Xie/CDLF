//
// Created by Sheen156 on 6/5/2018.
//

#ifndef RL_NONCONVEX_FCLAYER_H
#define RL_NONCONVEX_FCLAYER_H

#include "Layer.h"

// y = W*x+ b
// where y and b is m-D vector, y is output vector;
//       x is n-D input vector
//       W is m*n dimensional matrix
class FCLayer :  public Layer{
public:
    FCLayer(const long width, Layer* preLayer);
    ~FCLayer();

    long m_n; //input width
    long m_m; //output width

    DynamicMatrix<float>*  m_pW;
    DynamicVector<float>*  m_pBVector;
    DynamicMatrix<float>*  m_pdW;
    DynamicVector<float>*  m_pdBVector;

    virtual  void forward();
    virtual  void backward();
    virtual  void initialize(const string& initialMethod);



};


#endif //RL_NONCONVEX_FCLAYER_H
