//
// Created by Hui Xie on 7/16/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_TENSOR_H
#define RL_NONCONVEX_TENSOR_H

#include <vector>
#include <string>
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
    Tensor();
    Tensor(const vector<int>& dims);
    Tensor(const Tensor& other);
    Tensor& operator= (const Tensor& other);
    template <typename OtherValueType> Tensor& valueTypeConvertFrom(const Tensor<OtherValueType> &other);

    ~Tensor();

    void copyDataFrom(const ValueType* srcBuff, const int lengthInValueType, const int dstOffsetInValueType = 0);
    void copyDataFrom(const void* srcBuff, const int lengthInByte, const int dstOffsetInByte = 0);


    vector<int> getDims() const;
    vector<int> getDimsSpan() const;
    ValueType* getData() const;
    int getLength() const;

    void setDimsAndAllocateMem(const vector<int>& dims);

    inline ValueType& e(const vector<int>& index)const;
    inline ValueType& e(int index) const;
    inline ValueType& e(int i, int j) const;
    inline ValueType& e(int i, int j, int k) const;
    inline ValueType& e(int i, int j, int k, int l)const;
    inline ValueType& e(int i, int j, int k, int l, int m)const;
    inline ValueType& e(int i, int j, int k, int l, int m, int n)const;
    inline ValueType& e(int i, int j, int k, int l, int m, int n, int o)const;
    inline ValueType& operator[] (int index) const;
    inline ValueType& operator() (int index) const;
    inline ValueType& operator() (int i, int j) const;
    inline ValueType& operator() (int i, int j, int k) const;
    inline ValueType& operator() (int i, int j, int k, int l) const;
    inline ValueType& operator() (int i, int j, int k, int l, int m) const;
    inline ValueType& operator() (int i, int j, int k, int l, int m, int n) const;
    inline ValueType& operator() (int i, int j, int k, int l, int m, int n, int o) const;
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

    void zeroInitialize();
    void uniformInitialize(const ValueType x);

    float sum();
    float average();
    float variance();
    float max();
    float min();
    void getMinMax(ValueType& min, ValueType& max);
    int maxPosition();
    Tensor<unsigned char> getMaxPositionSubTensor();

    void save(const string& fullFilename, bool matrix2D=false);
    void load(const string& fullFilename, bool matrix2D=false);
    void print(bool fixWidth = false);

    Tensor ln();//natural logarithm
    Tensor expon();//exponential
    Tensor sign();
    Tensor hadamard(const Tensor& right); //element-wise product
    Tensor vectorize();
    float  dotProduct(const Tensor& right);
    Tensor reshape(vector<int> newDims);

    //before using subTensor function, user must allocate the memory of pTensor;
    void subTensorFromTopLeft(const vector<int>& tlIndex, Tensor* pTensor, const int stride =1) const ;
    void subTensorFromTopLeft(const int  offset, Tensor* pTensor, const int stride =1) const ;

    //dilute function will automatic allocate memory for pTensor
    void dilute(Tensor* & pTensor, const vector<int>& tensorSizeBeforeCollapse, const vector<int>& filterSize, const int stride) const;

    void putInBiggerTensor(Tensor* pBiggerTensor, const vector<int>& offsetVec, const int stride =1) const;

    // extractLowerDTensor will be repalced by slice, volume, fourDVolume
    Tensor column(const int index);
    Tensor row(const int index);
    Tensor slice(const int index);
    void volume(const int index, Tensor* & pTensor);
    void fourDVolume(const int index, Tensor* & pTensor);
    void extractLowerDTensor(const int index, Tensor* & pTensor);

    float conv(const Tensor& right, int nThreads=1) const; //convolution or cross-correlation
    Tensor& flip();
    float flipConv(const Tensor& right, int nThreads=1) const;

    inline vector<int> offset2Index(const int offset) const;
    inline vector<int> offset2Index(const vector<int>& dimsSpan, const int offset) const;


private:
    vector<int> m_dims;
    vector<int> m_dimsSpan; //express change of index leads how many data storage span.
    ValueType* m_data; //all data are stored in column major. Namely, the the index of 0th dimension varies most slowly

    void initializeMember();
    void allocateMem();
    void freeMem();
    void generateDimsSpan();
    inline int index2Offset(const vector<int>& index) const;
    inline int index2Offset(const vector<int>& dimsSpan, const vector<int>& index) const;

    void copyDataTo(Tensor* pTensor, const int lengthInValueType, const int srcOffsetInValueType=0);

};

#include "../src/Tensor.hpp"


#endif //RL_NONCONVEX_TENSOR_H
