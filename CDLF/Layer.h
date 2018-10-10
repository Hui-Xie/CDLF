//
// Created by Hui Xie on 6/5/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_LAYER_H
#define RL_NONCONVEX_LAYER_H

#include <string>
//#include  <blaze/Math.h>
#include <list>
#include "Tensor.h"

#ifdef Use_GPU
  #include "LayerCuda.h"
#endif

//using namespace blaze;
using namespace std;

class Layer {
public:
    Layer(const int id, const string& name, const vector<long>& tensorSize);
    ~Layer();

    int m_id;
    string m_name;
    string m_type;

    string m_attribute; //Some supplement information for layer type

    Layer*  m_prevLayer;
    vector<long>  m_tensorSize;
    Tensor<float>* m_pYTensor;             //the output of this layer
    Tensor<float>* m_pdYTensor;          //dL/dy,where L is Loss

    virtual  void initialize(const string& initialMethod)=0;
    virtual  void zeroParaGradient() = 0;
    void zeroYTensor();
    void zeroDYTensor(); //ConvolutionLayer, MaxPoolLayer, ReLu all needs dX =0;
    virtual  void forward()=0;
    virtual  void backward(bool computeW)=0;
    virtual  void updateParameters(const float lr, const string& method, const int batchSize =1) = 0;
    virtual  long getNumParameters() = 0; // return the number of learning parameters


    virtual void addPreviousLayer(Layer* prevLayer);

    void printY();
    void printDY();

    void setAttribute(const string& attr);
    string getAttribute();

protected:
    void allocateYdYTensor();
    void freeYdYTensor();

private:
    //void printVector(Tensor<float>* vector);



};


#endif //RL_NONCONVEX_LAYER_H
