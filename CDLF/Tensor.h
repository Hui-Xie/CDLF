//
// Created by Hui Xie on 7/16/2018.
//

#ifndef RL_NONCONVEX_TENSOR_H
#define RL_NONCONVEX_TENSOR_H

#include <vector>
using namespace std;

// Column Vector is a Tensor({n,1}), and Tensor({n}) is incorrect express.;
// Row Vector is a Tensor({1,n});
// Now this Network has finished Tensor Reconstruction, and supported DAG network.


template<class ValueType>
class Tensor {
public:
    Tensor(const vector<int>& dims);
    Tensor(const Tensor& other);
    Tensor& operator= (const Tensor& other);
    ~Tensor();

    vector<int> getDims() const;
    ValueType* getData() const;
    long getLength() const;

    inline ValueType& e(const vector<int>& index)const;
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

    Tensor subTensorFromCenter(const vector<int>& centralIndex,const vector<int>& span, const int stride =1);
    Tensor subTensorFromTopLeft(const vector<int>& tfIndex,const vector<int>& span, const int stride =1);
    Tensor reduceDimension(const int index); //index is the indexOfLastDim
    ValueType conv(const Tensor& other) const; //convolution or cross-correlation
    Tensor& flip();

private:
    vector<int> m_dims;
    vector<long> m_dimsSpan; //express change of index leads how many data storage span.
    ValueType* m_data;

    void allocateMem();
    void freeMem();
    void generateDimsSpan();
    inline long index2Offset(const vector<int>& index) const;
};

#include "Tensor.hpp"


#endif //RL_NONCONVEX_TENSOR_H
