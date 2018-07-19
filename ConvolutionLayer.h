//
// Created by Sheen156 on 7/19/2018.
//

#ifndef RL_NONCONVEX_CONVOLUTIONLAYER_H
#define RL_NONCONVEX_CONVOLUTIONLAYER_H
#include "Layer.h"
#include <map>

/** Convolution layer
 * Y = W*X +b
 * where Y is the output at each voxel;
 *       W is the convolution filter, which is uniform in entire input;
 *       X is the receipt region of original input image;
 *       b is bias which is different at different voxel location
 *       * indicate convolution
 *
 *
 *
 *
 *
 * */


class ConvolutionLayer :  public Layer {
public:
    ConvolutionLayer(const int id, const string& name, const vector<int>& tensorSize, list<Layer*>& prevLayers);
    ConvolutionLayer(const int id, const string& name, const vector<int>& tensorSize, Layer* prevLayer);
    ~ConvolutionLayer();

    struct LayerPara{
        Tensor<float>*  m_pW;
        Tensor<float>*  m_pBTensor;
        Tensor<float>*  m_pdW;
        Tensor<float>*  m_pdBTensor;
    };
    map<Layer*, LayerPara> m_layerParaMap;

    void constructLayerParaMap(list<Layer*>& prevLayers);


    virtual  void initialize(const string& initialMethod);
    virtual  void forward();
    virtual  void backward();
    virtual  void updateParameters(const float lr, const string& method);

private:
    int m_stride;




};


#endif //RL_NONCONVEX_CONVOLUTIONLAYER_H
