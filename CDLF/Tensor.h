//
// Created by Hui Xie on 7/16/2018.
//

#ifndef RL_NONCONVEX_TENSOR_H
#define RL_NONCONVEX_TENSOR_H

#include <vector>
using namespace std;

// Column Vector is a Tensor({n,1}), and Tensor({n}) is incorrect express.;
// Row Vector is a Tensor({1,n}); Namely, there is no 1D tensor.
// For 3D Tensor, the dimensional order is nSlice*Height*Width;
// For 4D Tensor, the dimensional order is nVolume*nSlice*Height*Width;
// For 5D Tensor, the dimensional order is n4D*nVolume*nSlice*Height*Width;

template<class ValueType>
class Tensor {
public:
    Tensor(const vector<long>& dims);
    Tensor(const Tensor& other);
    Tensor& operator= (const Tensor& other);
    ~Tensor();

    vector<long> getDims() const;
    ValueType* getData() const;
    long getLength() const;

    inline ValueType& e(const vector<long>& index)const;
    inline ValueType& e(long index) const;
    inline ValueType& e(long i, long j) const;
    inline ValueType& e(long i, long j, long k) const;
    inline ValueType& e(long i, long j, long k, long l)const;
    inline ValueType& e(long i, long j, long k, long l, long m)const;
    inline ValueType& operator[] (long index) const;
    inline ValueType& operator() (long index) const;
    inline ValueType& operator() (long i, long j) const;
    inline ValueType& operator() (long i, long j, long k) const;
    inline ValueType& operator() (long i, long j, long k, long l) const;
    inline ValueType& operator() (long i, long j, long k, long l, long m) const;
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

    Tensor& operator+= (const float right);
    Tensor& operator-= (const float right);
    Tensor& operator*= (const float factor);
    Tensor& operator/= (const float divisor);

    void printElements();
    void zeroInitialize();

    ValueType sum();
    ValueType average();
    ValueType variance();
    ValueType max();

    Tensor subTensorFromCenter(const vector<long>& centralIndex,const vector<long>& span, const int stride =1);
    Tensor subTensorFromTopLeft(const vector<long>& tfIndex,const vector<long>& span, const int stride =1);

    // extractLowerDTensor will be repalced by slice, volume, fourDVolume
    Tensor column(const int index);
    Tensor row(const int index);
    Tensor slice(const int index);
    Tensor volume(const int index);
    Tensor fourDVolume(const int index);
    Tensor extractLowerDTensor(const int index);

    ValueType conv(const Tensor& other) const; //convolution or cross-correlation
    Tensor& flip();

private:
    vector<long> m_dims;
    vector<long> m_dimsSpan; //express change of index leads how many data storage span.
    ValueType* m_data;

    void allocateMem();
    void freeMem();
    void generateDimsSpan();
    inline long index2Offset(const vector<long>& index) const;
    void copyDataTo(Tensor* pTensor, const long offset, const long length);
};

#include "Tensor.hpp"


#endif //RL_NONCONVEX_TENSOR_H
