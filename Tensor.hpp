//
// Created by Sheen156 on 7/16/2018.
//

#include "Tensor.h"
#include <iostream>
#include "Tools.h"
#include <assert.h>
#include <cstring>  //for memcpy function
#include <cmath> //for pow()

template<class ValueType>
Tensor<ValueType>::Tensor(const vector<int>& dims){
   m_dims = dims;
   m_data = nullptr;
   allocateMem();
}

template<class ValueType>
Tensor<ValueType>::Tensor(const Tensor& other){
    if (this != &other){
        m_data = nullptr;
        *this = other;
    }
}

template<class ValueType>
void Tensor<ValueType>::zeroInitialize(){
    long N= getLength();
    for(long i=0; i<N;++i){
        e(i) = 0;
    }
}


template<class ValueType>
Tensor<ValueType>& Tensor<ValueType>::operator= (const Tensor<ValueType>& other){
    if (this != &other) {
        long length = other.getLength();
        if (!sameVector(m_dims, other.getDims())){
            freeMem();
            m_dims = other.getDims();
            allocateMem();
        }
        memcpy(m_data, other.getData(), length*sizeof(ValueType));
    }
    return *this;
}

template<class ValueType>
Tensor<ValueType>::~Tensor(){
   freeMem();
}

template<class ValueType>
vector<int> Tensor<ValueType>::getDims()const {
   return m_dims;
}

template<class ValueType>
ValueType* Tensor<ValueType>::getData() const{
    return m_data;
}

template<class ValueType>
long Tensor<ValueType>::getLength() const{
    unsigned long length=1;
    int dim = m_dims.size();
    for(int i =0; i< dim; ++i){
        length *= m_dims[i];
    }
    return length;
}

template<class ValueType>
void  Tensor<ValueType>::allocateMem(){
   if (nullptr != m_data){
       delete[] m_data;
   }
   m_data = new ValueType[getLength()];
}

template<class ValueType>
void  Tensor<ValueType>::freeMem(){
   if (nullptr != m_data){
      delete[] m_data;
      m_data = nullptr;
   }
}


template<class ValueType>
ValueType& Tensor<ValueType>::e(const vector<int>& index) const{
    assert(index.size() == m_dims.size());
    int dim = m_dims.size();
    unsigned  long pos = 0;
    for (int i=0; i< dim-1; ++i){
        long iSpan = 1;
        for (int j=i+1; j<dim; ++j){
            iSpan *= m_dims[j];
        }
        pos += iSpan*index[i];
    }
    pos += index[dim-1];
    return m_data[pos];
}

template<class ValueType>
ValueType& Tensor<ValueType>::e(long index) const{
    return m_data[index];
}

template<class ValueType>
ValueType& Tensor<ValueType>::operator[] (long index){
    return m_data[index];
}

template<class ValueType>
ValueType& Tensor<ValueType>::operator() (long index){
    return m_data[index];
}

template<class ValueType>
ValueType& Tensor<ValueType>::operator() (long i, long j){
    assert(2 == m_dims.size());
    return m_data[i*m_dims[1]+j];
}

template<class ValueType>
ValueType& Tensor<ValueType>::operator() (long i, long j, long k){
    assert(3 == m_dims.size());
    return m_data[i*m_dims[1]*m_dims[2]+j*m_dims[2]+k];
}

template<class ValueType>
ValueType& Tensor<ValueType>::operator() (long i, long j, long k, long l){
    assert(4 == m_dims.size());
    return m_data[i*m_dims[1]*m_dims[2]*m_dims[3]+j*m_dims[2]*m_dims[3]+k*m_dims[3]+l];
}

template<class ValueType>
ValueType& Tensor<ValueType>::operator() (long i, long j, long k, long l, long m){
    assert(5 == m_dims.size());
    return m_data[i*m_dims[1]*m_dims[2]*m_dims[3]*m_dims[4]+j*m_dims[2]*m_dims[3]*m_dims[4]+k*m_dims[3]*m_dims[4]
                   +l*m_dims[4] +m];
}

// transpose operation only supports 2D matrix
template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::transpose(){
    vector<int> newDims = reverseVector(m_dims);
    Tensor tensor (newDims);
    int dim = m_dims.size();
    assert(dim ==2 );
    for (int i=0; i<newDims[0]; ++i){
        for (int j=0; j< newDims[1];++j){
            tensor.e({i,j}) = e({j,i});
        }
    }
    return tensor;
}



template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator* (const Tensor<ValueType>& other){
    int thisDim = m_dims.size();
    vector<int> otherDims = other.getDims();
    int otherDim = otherDims.size();
    assert (m_dims[thisDim-1] == otherDims[0]);
    assert (2 == thisDim && 2 == otherDim);

    vector<int> newDims{m_dims[0], otherDims[1]};
    Tensor tensor (newDims);
    for (int i=0; i<newDims[0]; ++i){
        for (int j=0; j< newDims[1];++j){
            ValueType value =0;
            for (int k=0; k< m_dims[1]; ++k){
                value += e({i,k})*other.e({k,j});
            }
            tensor.e({i,j}) = value;
        }
    }
    return tensor;
}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator+ (const float other){
    Tensor tensor (m_dims);
    long N = tensor.getLength();
    for (long i=0; i<N; ++i){
        tensor.e(i) =e(i)+ other;
    }
    return tensor;
}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator- (const float other){
    Tensor tensor (m_dims);
    long N = tensor.getLength();
    for (long i=0; i<N; ++i){
        tensor.e(i) =e(i)- other;
    }
    return tensor;
}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator* (const float factor){
    Tensor tensor (m_dims);
    long N = tensor.getLength();
    for (long i=0; i<N; ++i){
         tensor.e(i) =e(i)* factor;
    }
    return tensor;
}



template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator+ (const Tensor<ValueType>& other){
   assert(sameVector(m_dims, other.getDims()));
   Tensor tensor (m_dims);
   long N = getLength();
   for (long i=0; i<N; ++i){
       tensor.e(i) = e(i) + other.e(i);
    }
    return tensor;

}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator- (const Tensor<ValueType>& other){
    assert(sameVector(m_dims, other.getDims()));
    Tensor tensor (m_dims);
    long N = getLength();
    for (long i=0; i<N; ++i){
        tensor.e(i) = e(i) - other.e(i);
    }
    return tensor;
}
template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator/ (const float divisor){
    if (0 == divisor){
        return *this;
    }
    else{
        Tensor tensor (m_dims);
        long N = getLength();
        for (long i=0; i<N; ++i){
            tensor.e(i) = e(i)/divisor;
        }
        return tensor;
    }
}

template<class ValueType>
Tensor<ValueType>& Tensor<ValueType>::operator+= (const Tensor& right){
    assert(sameVector(m_dims, right.getDims()));
    long N = getLength();
    for (long i=0; i<N; ++i){
        e(i) += right.e(i);
    }
    return *this;
}

template<class ValueType>
Tensor<ValueType>& Tensor<ValueType>::operator-= (const Tensor& right){
    assert(sameVector(m_dims, right.getDims()));
    long N = getLength();
    for (long i=0; i<N; ++i){
        e(i) -= right.e(i);
    }
    return *this;
}

template<class ValueType>
Tensor<ValueType>& Tensor<ValueType>::operator+= (const float right){
    long N = getLength();
    for (long i=0; i<N; ++i){
        e(i) += right;
    }
    return *this;
}

template<class ValueType>
Tensor<ValueType>& Tensor<ValueType>::operator-= (const float right){
    long N = getLength();
    for (long i=0; i<N; ++i){
        e(i) -= right;
    }
    return *this;
}

template<class ValueType>
Tensor<ValueType>& Tensor<ValueType>::operator*= (const float factor){
    long N = getLength();
    for (long i=0; i<N; ++i){
        e(i) *= factor;
    }
    return *this;
}

template<class ValueType>
Tensor<ValueType>& Tensor<ValueType>::operator/= (const float divisor){
    if (0 != divisor) {
        long N = getLength();
        for (long i = 0; i < N; ++i) {
            e(i) /= divisor;
        }
    }
    return *this;
}

template<class ValueType>
void Tensor<ValueType>::printElements(){
    assert(2 == m_dims.size());
    for (int i=0; i< m_dims[0];++i){
        for(int j=0; j<m_dims[1];++j) {
            cout << e({i, j}) << "  ";
        }
        cout<<endl;
    }
}

template<class ValueType>
float Tensor<ValueType>::sum(){
    long N = getLength();
    float sum = 0.0;
    for(long i=0; i<N; ++i){
        sum += e(i);
    }
    return sum;
}

template<class ValueType>
float Tensor<ValueType>::average(){
    long N = getLength();
    if (0 == N) return 0;
    else {
        return sum()/N;
    }
}

template<class ValueType>
float Tensor<ValueType>::variance(){
    float mu = average();
    long N = getLength();
    if (1 == N || 0 == N) return 0;
    float sum = 0.0;
    for (long i = 0; i < N; ++i) {
        sum += pow((e(i) - mu), 2);
    }
    return sum / (N - 1); // population variance
}
