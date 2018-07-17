//
// Created by Sheen156 on 7/16/2018.
//

#ifndef RL_NONCONVEX_TENSOR_H
#define RL_NONCONVEX_TENSOR_H

#include <vector>
using namespace std;

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

    float sum();
    float average();
    float variance();

private:
    vector<int> m_dims;
    ValueType* m_data;
    void allocateMem();
    void freeMem();

};

#include "Tensor.hpp"


#endif //RL_NONCONVEX_TENSOR_H
