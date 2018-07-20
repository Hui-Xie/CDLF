//
// Created by Sheen156 on 7/16/2018.
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

    ValueType& e(const vector<int>& index)const;
    ValueType& e(long index) const;
    ValueType& e(long i, long j) const;
    ValueType& e(long i, long j, long k) const;
    ValueType& e(long i, long j, long k, long l)const;
    ValueType& e(long i, long j, long k, long l, long m)const;
    ValueType& operator[] (long index) const;
    ValueType& operator() (long index) const;
    ValueType& operator() (long i, long j) const;
    ValueType& operator() (long i, long j, long k) const;
    ValueType& operator() (long i, long j, long k, long l) const;
    ValueType& operator() (long i, long j, long k, long l, long m) const;
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

    float sum();
    float average();
    float variance();

    Tensor subTensor(const vector<int>& centralIndex,const vector<int>& span);
    Tensor reduceDimension(const int index); //index is the indexOfLastDim

private:
    vector<int> m_dims;
    vector<long> m_dimsSpan; //express change of index leads how many data storage span.
    ValueType* m_data;

    void allocateMem();
    void freeMem();
    void generateDimsSpan();
    long index2Offset(const vector<int>& index) const;
};

#include "Tensor.hpp"


#endif //RL_NONCONVEX_TENSOR_H
