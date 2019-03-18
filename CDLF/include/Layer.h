//
// Created by Hui Xie on 6/5/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_LAYER_H
#define CDLF_FRAME_LAYER_H

#include <string>
//#include  <blaze/Math.h>
#include <list>
#include "Tensor.h"
#include <cstdio>
#include "FileTools.h"

#ifdef Use_GPU
  #include "LayerCuda.h"
#endif

/* Learning Rate thinking:
 *
 * For every learningRate optimizer, the majority of learning rates fail to train the model.
 * There is a valley shape for each optimizer: too low a learning rate never progresses, too high a learning rate causes instability and never converges.
 *    In between there is a band of “just right” learning rates that successfully train.
 * There is no learning rate that works for all optimizers.
 * Learning rate can affect training time by an order of magnitude.
 * ---- abstracted from: https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2
 *
 *
 *  Smith writes, the main assumption behind the rationale for a cyclical learning rate
 *     is "that increasing the learning rate might have a short term negative effect and yet achieve a longer term beneficial effect."
 *
 * increasing the learning rate can also allow for "more rapid traversal of saddle point plateaus."
 *
 * A similar approach cyclic approach is known as stochastic gradient descent with warm restarts where an aggressive annealing schedule
 * is combined with periodic "restarts" to the original starting learning rate.
 * By drastically increasing the learning rate at each restart, we can essentially exit a local minima and continue exploring the loss landscape.
 * ---- abstracted from: https://www.jeremyjordan.me/nn-learning-rate/
 *
 * */



//using namespace blaze;
using namespace std;

class Layer {
public:
    Layer(const int id, const string& name, const vector<int>& tensorSize);
    ~Layer();

    int m_id; // id >0, 0 means null;
    string m_name;
    string m_type;

    string m_attribute; //Some supplement information for layer type

    Layer*  m_prevLayer;
    vector<int>  m_tensorSize;
    Tensor<float>* m_pYTensor;             //the output of this layer
    Tensor<float>* m_pdYTensor;          //dL/dy,where L is Loss

    virtual  void initialize(const string& initialMethod)=0;
    virtual  void zeroParaGradient() = 0;
    void zeroYTensor();
    void zeroDYTensor(); //ConvolutionLayer, MaxPoolLayer, ReLu all needs dX =0;
    virtual  void forward()=0;
    virtual  void backward(bool computeW, bool computeX = true)=0;
    virtual  void updateParameters(const float lr, const string& method, const int batchSize =1) = 0;
    virtual  int getNumParameters() = 0; // return the number of learning parameters

    // save and load methods are only for learning parameters
    virtual  void save(const string& netDir)=0;
    virtual  void load(const string& netDir)=0;

    virtual  void saveStructLine(FILE* pFile)=0;
    virtual  void printStruct() = 0;


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


#endif //CDLF_FRAME_LAYER_H
