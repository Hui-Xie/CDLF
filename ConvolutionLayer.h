//
// Created by Sheen156 on 7/19/2018.
//

#ifndef RL_NONCONVEX_CONVOLUTIONLAYER_H
#define RL_NONCONVEX_CONVOLUTIONLAYER_H
#include "Layer.h"
#include <map>

/** Convolution layer
 * Y = W*X
 * where Y is the output at each voxel;
 *       W is the convolution filter, which is uniform in entire input;
 *       X is the receipt region of original input image;
 *       * indicate convolution
 *
 * Notes:
 * 1  currently only supports one previous layer
 * 2  in convolution layer, we do not consider bias, as there is a separate BiasLayer for use;
 * 3  Size changes: |Y| = |X|-|W|+1, in their different dimension;
 * 4  the dimension of tensorSize of filter = dimension of tensorSize of X +1;
 *    the adding dimension expresses the number of filter;
 *
 *
 *
 *
 *
 *
 *
 *
 * */


class ConvolutionLayer :  public Layer {
public:
    ConvolutionLayer(const int id, const string& name, const vector<int>& filterSize, Layer* prevLayer,
                     const int numFilters=1, const int stride=1);
    ~ConvolutionLayer();

    Tensor<float>*  m_pW[];
    Tensor<float>*  m_pdW[];
    Tensor<float>*  m_expandDy;

    void constructFiltersAndY();


    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward();
    virtual  void updateParameters(const float lr, const string& method);

private:
    int m_stride;
    int m_OneFilterN;
    vector<int> m_filterSize;
    int m_numFilters;

    bool checkFilterSize(const vector<int>& filterSize, Layer* prevLayer);
    void expandDyTensor(Tensor<float>* pdY);
    void freeExpandDy();
    void computeDW(Tensor<float>* pdY, Tensor<float>* pdW);
    void computeDX(Tensor<float>* pdY, Tensor<float>* pW);//Note: dx need to accumulate along filters

};


#endif //RL_NONCONVEX_CONVOLUTIONLAYER_H
