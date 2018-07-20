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
   generateDimsSpan();
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
            generateDimsSpan();
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
void Tensor<ValueType>::generateDimsSpan(){
    int N = m_dims.size();
    m_dimsSpan.clear();
    for (int i=0; i<N; ++i){
        long span = 1;
        for(int j=i+1; j<N; ++j){
            span *= m_dims[j];
        }
        m_dimsSpan.push_back(span);
    }
}

template<class ValueType>
long Tensor<ValueType>::index2Offset(const vector<int>& index)const{
    int N = index.size();
    long offset =0;
    for (int i=0; i<N; ++i){
        offset += index[i]*m_dimsSpan[i];
    }
    return offset;
}

template<class ValueType>
ValueType& Tensor<ValueType>::e(const vector<int>& index) const{
    assert(index.size() == m_dims.size());
    return m_data[index2Offset(index)];
}

template<class ValueType>
ValueType& Tensor<ValueType>::e(long index) const{
    return m_data[index];
}

template<class ValueType>
ValueType& Tensor<ValueType>::e(long i, long j) const{
    assert(2 == m_dims.size());
    return m_data[i*m_dimsSpan[0]+j*m_dimsSpan[1]];
}

template<class ValueType>
ValueType& Tensor<ValueType>::e(long i, long j, long k) const{
    assert(3 == m_dims.size());
    return m_data[i*m_dimsSpan[0]+j*m_dimsSpan[1]+k*m_dimsSpan[2]];
}

template<class ValueType>
ValueType& Tensor<ValueType>::e(long i, long j, long k, long l)const{
    assert(4 == m_dims.size());
    return m_data[i*m_dimsSpan[0]+j*m_dimsSpan[1]+k*m_dimsSpan[2]+l*m_dimsSpan[3]];
}

template<class ValueType>
ValueType& Tensor<ValueType>::e(long i, long j, long k, long l, long m)const{
    assert(5 == m_dims.size());
    return m_data[i*m_dimsSpan[0]+j*m_dimsSpan[1]+k*m_dimsSpan[2]+l*m_dimsSpan[3] +m*m_dimsSpan[4]];
}

template<class ValueType>
ValueType& Tensor<ValueType>::operator[] (long index) const {
    return m_data[index];
}

template<class ValueType>
ValueType& Tensor<ValueType>::operator() (long index) const {
    return m_data[index];
}

template<class ValueType>
ValueType& Tensor<ValueType>::operator() (long i, long j) const {
    assert(2 == m_dims.size());
    return m_data[i*m_dimsSpan[0]+j*m_dimsSpan[1]];
}

template<class ValueType>
ValueType& Tensor<ValueType>::operator() (long i, long j, long k) const {
    assert(3 == m_dims.size());
    return m_data[i*m_dimsSpan[0]+j*m_dimsSpan[1]+k*m_dimsSpan[2]];
}

template<class ValueType>
ValueType& Tensor<ValueType>::operator() (long i, long j, long k, long l) const {
    assert(4 == m_dims.size());
    return m_data[i*m_dimsSpan[0]+j*m_dimsSpan[1]+k*m_dimsSpan[2]+l*m_dimsSpan[3]];
}

template<class ValueType>
ValueType& Tensor<ValueType>::operator() (long i, long j, long k, long l, long m) const {
    assert(5 == m_dims.size());
    return m_data[i*m_dimsSpan[0]+j*m_dimsSpan[1]+k*m_dimsSpan[2]+l*m_dimsSpan[3] +m*m_dimsSpan[4]];
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


//Only support 2 dimensional tensor
template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator* (const Tensor<ValueType>& other){
    int thisDim = m_dims.size();
    vector<int> otherDims = other.getDims();
    int otherDim = otherDims.size();
    if  (m_dims[thisDim-1] != otherDims[0]){
        cout<<"Error: Tensor product has un-matching dimension."<<endl;
        return *this;
    }

    if (2 == thisDim && 2 == otherDim){
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
    else {
        cout <<"Error: Tensor product only support 2D tensor."<<endl;
        return *this;
     }

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

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::subTensor(const vector<int>& centralIndex,const vector<int>& span){
    Tensor tensor (span);
    int N = span.size();
    vector<int> halfSpan; //also the central voxel in the tensor(span)
    for(int i =0; i<N; ++i){
        halfSpan.push_back(span[i]/2);
    }

    if (2 == N){
        for (int i=-halfSpan[0]; i<=halfSpan[0]; ++i){
            for (int j=-halfSpan[1];j<=halfSpan[1];++j){
                tensor(halfSpan[0]+i, halfSpan[1]+j) = e(centralIndex[0]+i, centralIndex[1]+j);
            }
        }
    }
    else if (3 == N){
        for (int i=-halfSpan[0]; i<=halfSpan[0]; ++i){
            for (int j=-halfSpan[1];j<=halfSpan[1];++j){
                for (int k=-halfSpan[2];k<=halfSpan[2];++k){
                    tensor(halfSpan[0]+i, halfSpan[1]+j, halfSpan[2]+k)
                       = e(centralIndex[0]+i, centralIndex[1]+j, centralIndex[2]+k);
                }
             }
        }
    }
    else if (4 ==N){
        for (int i=-halfSpan[0]; i<=halfSpan[0]; ++i){
            for (int j=-halfSpan[1];j<=halfSpan[1];++j){
                for (int k=-halfSpan[2];k<=halfSpan[2];++k){
                    for (int l=-halfSpan[3];l<=halfSpan[3];++l){
                        tensor(halfSpan[0]+i, halfSpan[1]+j, halfSpan[2]+k,halfSpan[3]+l)
                                = e(centralIndex[0]+i, centralIndex[1]+j, centralIndex[2]+k, centralIndex[3]+l);
                    }
                }
            }
        }
    }
    else if (5 == N){
        for (int i=-halfSpan[0]; i<=halfSpan[0]; ++i){
            for (int j=-halfSpan[1];j<=halfSpan[1];++j){
                for (int k=-halfSpan[2];k<=halfSpan[2];++k){
                    for (int l=-halfSpan[3];l<=halfSpan[3];++l){
                        for (int m=-halfSpan[4];m<=halfSpan[4]; ++m){
                            tensor(halfSpan[0]+i, halfSpan[1]+j, halfSpan[2]+k,halfSpan[3]+l,halfSpan[4]+m )
                               = e(centralIndex[0]+i, centralIndex[1]+j, centralIndex[2]+k, centralIndex[3]+l,centralIndex[4]+m);
                        }
                     }
                }
            }
        }
    }
    else{
        cout<<"Error: currently do not support 6 and higher dimension in tensor."<<endl;
    }
    return tensor;

}
template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::reduceDimension(const int index){
    const int oldN= m_dims.size();
    vector<int> newDims;
    if (2 == oldN){
         newDims.push_back(m_dims[0]);
         newDims.push_back(1);
    }
    else{
         newDims = m_dims;
         newDims.erase(newDims.end()-1);
    }
    Tensor tensor(newDims);
    const int newN = newDims.size();

    if (2 == newN && 2 == oldN){
       for (int i=0; i< newDims[0]; ++i){
           tensor[i] = e(i,index);
       }
    }
    else if (2 == newN && 3 == oldN){
        for (int i=0; i< newDims[0]; ++i){
            for(int j=0; j<newDims[1];++j){
                tensor(i,j) = e(i,j,index);
            }
        }
     }
    else if (3 == newN){
        for (int i=0; i< newDims[0]; ++i){
            for(int j=0; j<newDims[1];++j){
                for (int k=0; k<newDims[2];++k){
                     tensor(i,j,k) = e(i,j,k,index);
                }
            }
        }
    }
    else if (4 == newN){
        for (int i=0; i< newDims[0]; ++i){
            for(int j=0; j<newDims[1];++j){
                for (int k=0; k<newDims[2];++k){
                    for(int l=0; l<newDims[3];++l){
                         tensor(i,j,k,l) = e(i,j,k,l,index);
                    }
                }
            }
        }
    }
    else{
        cout<<"Error: we only support 5D Tensor at most"<<endl;
    }
    return tensor;

}


//convolution or cross-correlation
template<class ValueType>
ValueType Tensor<ValueType>::conv(const Tensor &other) const{
    assert(sameVector(m_dims, other.getDims()));
    long N = getLength();
    ValueType sum = 0;
    for (long i = 0; i < N; ++i) {
        sum += e(i) * other[i];
    }
    return sum;

}
