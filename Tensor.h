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
    ~Tensor();

    vector<int> getDims() const;
    ValueType* getData();
    long getLength() const;

    ValueType& e(const vector<int>& index);
    ValueType& e(long index);
    Tensor transpose();
    Tensor operator* (const Tensor& other);
    Tensor& operator= (const Tensor& other);
    Tensor& operator+ (const Tensor& other);
    Tensor& operator- (const Tensor& other);
    void printElements();





private:
    vector<int> m_dims;
    ValueType* m_data;
    void allocateMem();
    void freeMem();

};

#include "Tensor.hpp"


#endif //RL_NONCONVEX_TENSOR_H
