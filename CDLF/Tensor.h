//
// Created by Hui Xie on 7/16/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_TENSOR_H
#define RL_NONCONVEX_TENSOR_H

#include <vector>
using namespace std;

// Column Vector is a Tensor({n,1}), and Tensor({n}) is incorrect expression;
// Row Vector is a Tensor({1,n}); Namely, there is no 1D tensor.
// For 3D Tensor, the dimensional order is nSlice*Height*Width;
// For 4D Tensor, the dimensional order is nVolume*nSlice*Height*Width;
// For 5D Tensor, the dimensional order is n4D*nVolume*nSlice*Height*Width;
// This Tensor supports 7D tensor maximal.

template<class ValueType>
class Tensor {
public:
    Tensor(const vector<long>& dims);
    Tensor(const Tensor& other);
    Tensor& operator= (const Tensor& other);
    ~Tensor();

    void copyDataFrom(void* buff, const long numBytes);


    vector<long> getDims() const;
    ValueType* getData() const;
    long getLength() const;

    inline ValueType& e(const vector<long>& index)const;
    inline ValueType& e(long index) const;
    inline ValueType& e(long i, long j) const;
    inline ValueType& e(long i, long j, long k) const;
    inline ValueType& e(long i, long j, long k, long l)const;
    inline ValueType& e(long i, long j, long k, long l, long m)const;
    inline ValueType& e(long i, long j, long k, long l, long m, long n)const;
    inline ValueType& e(long i, long j, long k, long l, long m, long n, long o)const;
    inline ValueType& operator[] (long index) const;
    inline ValueType& operator() (long index) const;
    inline ValueType& operator() (long i, long j) const;
    inline ValueType& operator() (long i, long j, long k) const;
    inline ValueType& operator() (long i, long j, long k, long l) const;
    inline ValueType& operator() (long i, long j, long k, long l, long m) const;
    inline ValueType& operator() (long i, long j, long k, long l, long m, long n) const;
    inline ValueType& operator() (long i, long j, long k, long l, long m, long n, long o) const;
    Tensor transpose();


    Tensor operator+ (const Tensor& other);
    Tensor operator- (const Tensor& other);
    Tensor operator* (const Tensor& other);

    Tensor operator+ (const float other);
    Tensor operator- (const float other);
    Tensor operator* (const float factor);
    Tensor operator/ (const float divisor);

    Tensor& operator+= (const Tensor& right);
    Tensor& operator-= (const Tensor& right);
    bool    operator== (const Tensor& right);
    bool    operator!= (const Tensor& right);

    Tensor& operator+= (const float right);
    Tensor& operator-= (const float right);
    Tensor& operator*= (const float factor);
    Tensor& operator/= (const float divisor);


    void printElements(bool fixWidth = false);
    void zeroInitialize();
    void uniformInitialize(const ValueType x);

    float sum();
    float average();
    float variance();
    float max();
    float min();
    long maxPosition();
    Tensor<unsigned char> getMaxPositionSubTensor();


    Tensor ln();//natural logarithm
    Tensor hadamard(const Tensor& right); //element-wise product
    Tensor vectorize();
    float  dotProduct(const Tensor& right);
    Tensor reshape(vector<long> newDims);


    void subTensorFromCenter(const vector<long>& centralIndex,const vector<long>& span, Tensor* & pTensor, const int stride =1);
    void subTensorFromTopLeft(const vector<long>& tfIndex,const vector<long>& span, Tensor* & pTensor, const int stride =1);

    // extractLowerDTensor will be repalced by slice, volume, fourDVolume
    Tensor column(const int index);
    Tensor row(const int index);
    Tensor slice(const int index);
    void volume(const int index, Tensor* & pTensor);
    void fourDVolume(const int index, Tensor* & pTensor);
    void extractLowerDTensor(const int index, Tensor* & pTensor);

    ValueType conv(const Tensor& other) const; //convolution or cross-correlation
    Tensor& flip();

    inline vector<long> offset2Index(const long offset) const;


private:
    vector<long> m_dims;
    vector<long> m_dimsSpan; //express change of index leads how many data storage span.
    ValueType* m_data; //all data are stored in column major. Namely, the the index of 0th dimension varies most slowly

    void allocateMem();
    void freeMem();
    void generateDimsSpan();
    inline long index2Offset(const vector<long>& index) const;

    void copyDataTo(Tensor* pTensor, const long offset, const long length);

};

#include "Tensor.hpp"


#endif //RL_NONCONVEX_TENSOR_H
