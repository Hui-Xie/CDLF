//
// Created by Sheen156 on 6/5/2018.
//

#ifndef RL_NONCONVEX_FCLAYER_H
#define RL_NONCONVEX_FCLAYER_H

#include "Layer.h"
#include <map>


// y = W*x+ b
// where y and b is m-D vector, y is output vector;
//       x is n-D input vector
//       W is m*n dimensional matrix
class FCLayer :  public Layer{
public:
    FCLayer(const int id, const string& name, const vector<int>& tensorSize, list<Layer*>& preLayers);
    ~FCLayer();

    long m_m; //output width
    struct LayerPara{
        long m_n; //input width
        Tensor<float>*  m_pW;
        Tensor<float>*  m_pBTensor;
        Tensor<float>*  m_pdW;
        Tensor<float>*  m_pdBTensor;
    };
    map<Layer*, LayerPara> m_layerParaMap;


    virtual  void initialize(const string& initialMethod);
    virtual  void forward();
    virtual  void backward();
    virtual  void updateParameters(const float lr, const string& method);

    void printWandBVector();
    void printdWanddBVector();
};


#endif //RL_NONCONVEX_FCLAYER_H
