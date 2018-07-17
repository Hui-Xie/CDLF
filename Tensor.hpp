//
// Created by Sheen156 on 7/16/2018.
//

#include "Tensor.h"
#include <iostream>
#include "Tools.h"
#include <assert.h>
#include <cstring>  //for memcpy function

template<class ValueType>
Tensor<ValueType>::Tensor(const vector<int>& dims){
   m_dims = dims;
   m_data = nullptr;
   allocateMem();
}

template<class ValueType>
Tensor<ValueType>::Tensor(const Tensor& other){
    if (this != &other){
        *this = other;
    }
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
ValueType& Tensor<ValueType>::e(long index){
    return m_data[index];
}

template<class ValueType>
ValueType& Tensor<ValueType>::e(const vector<int>& index){
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

// transpose operation only supports 2D matrix
template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::transpose(){
    vector<int> newDims = reverseVector(m_dims);
    Tensor tensor (newDims);
    int dim = m_dims.size();
    assert(dim ==2 );
    for (int i=0; i<newDims[0]; ++i){
        for (int j=0; j< newDims[1];++j){
            tensor.e(i*newDims[1]+j) = e(i*m_dims[1]+j);
        }
    }
    return tensor;
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
Tensor<ValueType> Tensor<ValueType>::operator* (const Tensor& other){
    int thisDim = m_dims.size();
    vector<int> otherDims = other.getDims();
    int otherDim = otherDims.size();
    if (m_dims[thisDim-1] != otherDims[otherDim -1]){
        cout<<"Error: tensor product should has matching dimension."<<endl;
        return nullptr;
    }
    else {


    }


}



template<class ValueType>
Tensor<ValueType>& Tensor<ValueType>::operator+ (const Tensor& other){

}

template<class ValueType>
Tensor<ValueType>& Tensor<ValueType>::operator- (const Tensor& other){

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
