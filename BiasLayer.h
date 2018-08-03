//
// Created by Sheen156 on 7/19/2018.
//

#ifndef RL_NONCONVEX_BIASLAYER_H
#define RL_NONCONVEX_BIASLAYER_H
#include "Layer.h"

/*  Bias Layer
 *  generally put BiasLayer after ConvolutionLayer to indicate the bias of different voxel in previous layer;
 *  Bias Layer is used to express the spacial coordinates bias;
 *  if previous layer is 4D in size, then Bias Layer also indicates the bias of different convolution filter;
 *  Y = X + b
 *  where Y is the output of Bias Layer
 *        X is the input of the Bias Layer
 *        b is the learning parameter of Bias Layer, which is different at each voxel
 *  dL/dX = dL/dY    Where L is Loss
 *  dL/db = dL/dY
 * */

class BiasLayer : public Layer {
public:
    BiasLayer(const int id, const string& name, Layer* prevLayer);
    ~BiasLayer();

    Tensor<float>*  m_pBTensor;
    Tensor<float>*  m_pdBTensor;

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward();
    virtual  void updateParameters(const float lr, const string& method, const int batchSize=1);


};


#endif //RL_NONCONVEX_BIASLAYER_H
