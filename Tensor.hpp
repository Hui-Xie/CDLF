//
// Created by Sheen156 on 7/16/2018.
//

#include "Tensor.h"
#include <iostream>
#include "Tools.h"

template<class ValueType>
Tensor::Tensor(const vector<int>& dims){
   m_dims = dims;
   m_data = nullptr;
   allocateMem();
}

template<class ValueType>
Tensor::~Tensor(){
   freeMem();
}

template<class ValueType>
vector<int> Tensor::getDims()const {
   return m_dims;
}

template<class ValueType>
ValueType* Tensor::getData(){
    return m_data;
}

template<class ValueType>
long Tensor::getLength() const{
    unsigned long length=1;
    int dim = m_dims.size();
    for(int i =0; i< dim; ++i){
        length *= m_dims[i];
    }
    return length;
}

template<class ValueType>
void  Tensor::allocateMem(){
   if (nullptr != m_data){
       delete[] m_data;
   }
   m_data = new ValueType[getLength()];
}

template<class ValueType>
void  Tensor::freeMem(){
   if (nullptr != m_data){
      delete[] m_data;
      m_data = nullptr;
   }
}

template<class ValueType>
ValueType& Tensor::e(const vector<int>& index){
   if (index.size() != m_dims.size()){
      cout<<"Error: Tensor index and dims have different dimension."<<endl;
      return nullptr;
   }
   int dim = m_dims.size();
   unsigned  long pos = 0;
   for (int i=0; i< dim-1; ++i){
      pos += m_dims[i]*index[i];
   }
   pos += index[dim-1];
   return m_data[pos];
}

template<class ValueType>
Tensor Tensor::transpose(){
    Tensor<ValueType> tensor(m_dims);


    return tensor;

}

template<class ValueType>
Tensor& Tensor::operator= (const Tensor& other){
    if (this != &other) {
        long length = other.getLength();
        if (!sameVector(m_dims, other.getDims())){
             freeMem();
             m_dims = other.getDims();
             allocateMem();
        }
        std::copy(other.getData(), other.getData() + length, m_data);
    }
    return *this;
}


template<class ValueType>
Tensor Tensor::operator* (const Tensor& other){
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
Tensor& Tensor::operator+ (const Tensor& other){

}

template<class ValueType>
Tensor& Tensor::operator- (const Tensor& other){

}
