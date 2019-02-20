//
// Created by Hui Xie on 7/16/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "Tensor.h"
#include <iostream>
#include "Tools.h"
#include <assert.h>
#include <cstring>  //for memcpy function
#include <cmath> //for pow() and exp
#include <iomanip>      // std::setw
#include "CPUAttr.h"
#include <thread>
#include "ipp.h"

#ifdef Use_GPU
  #include <cuda_runtime.h>
  #include "TensorCuda.h"
  #include "GPUAttr.h"
#endif


#define checkIPP(expression)                               \
  {                                                          \
    IppStatus status = (expression);                     \
    if (status != ippStsNoErr) {                    \
      std::cerr << "Intel IPP Error: line " << __LINE__ << " of file: "<< __FILE__\
                << std::endl                                  \
                << ippGetStatusString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }



template<class ValueType>
void Tensor<ValueType>::initializeMember() {
    m_data = nullptr;
    m_dims.clear();
    m_dimsSpan.clear();
}

template<class ValueType>
Tensor<ValueType>::Tensor() {
    initializeMember();
}


template<class ValueType>
Tensor<ValueType>::Tensor(const vector<int> &dims) {
    initializeMember();
    m_dims = dims;
    if (1 == m_dims.size()) {
        m_dims.push_back(1);
    }
    generateDimsSpan();
    allocateMem();
}

template<class ValueType>
void Tensor<ValueType>::setDimsAndAllocateMem(const vector<int>& dims){
    freeMem();
    m_dims = dims;
    if (1 == m_dims.size()) {
        m_dims.push_back(1);
    }
    generateDimsSpan();
    allocateMem();
}

template<class ValueType>
Tensor<ValueType>::Tensor(const Tensor &other) {
    initializeMember();
    if (this != &other) {
        *this = other;
    }
}

template<class ValueType>
void Tensor<ValueType>::allocateMem() {
    if (getLength() > 0) {
        m_data = new ValueType[getLength()]; // for CPU
    }
}

template<class ValueType>
void Tensor<ValueType>::freeMem() {
    if (nullptr != m_data) {
        delete[] m_data;
    }
    m_data = nullptr;

}


template<class ValueType>
void Tensor<ValueType>::zeroInitialize() {
    int N = getLength();
    for (int i = 0; i < N; ++i) {
        e(i) = 0;
    }
}

template<class ValueType>
void Tensor<ValueType>::uniformInitialize(const ValueType x) {
    int N = getLength();
    for (int i = 0; i < N; ++i) {
        e(i) = x;
    }
}


template<class ValueType>
Tensor<ValueType>& Tensor<ValueType>::operator=(const Tensor &other) {
    if (this != &other) {
        freeMem();
        m_dims = other.getDims();
        generateDimsSpan();
        allocateMem();
        const int N = other.getLength();
        if (N > 0) {
        memcpy(m_data, other.getData(), N * sizeof(ValueType));
        }
    }
    return *this;
}

template<class ValueType>
template<typename OtherValueType>
Tensor<ValueType> &Tensor<ValueType>::valueTypeConvertFrom(const Tensor<OtherValueType> &other) {
    freeMem();
    m_dims = other.getDims();
    generateDimsSpan();
    allocateMem();
    const int N = other.getLength();
    for (int i = 0; i < N; ++i) {
        e(i) = (ValueType) other.e(i);
    }
    return *this;
}

template<class ValueType>
Tensor<ValueType>::~Tensor() {
    freeMem();
}

template<class ValueType>
void Tensor<ValueType>::copyDataFrom(const ValueType* srcBuff, const int lengthInValueType, const int dstOffsetInValueType) {
     memcpy(m_data + dstOffsetInValueType, srcBuff, lengthInValueType* sizeof(ValueType));
}

template<class ValueType>
void Tensor<ValueType>::copyDataFrom(const void* srcBuff, const int lengthInByte, const int dstOffsetInByte ){
    memcpy(m_data + dstOffsetInByte/sizeof(ValueType), srcBuff, lengthInByte);
}


template<class ValueType>
vector<int> Tensor<ValueType>::getDims() const {
    return m_dims;
}

template<class ValueType>
vector<int> Tensor<ValueType>::getDimsSpan() const{
    return m_dimsSpan;
}

template<class ValueType>
ValueType *Tensor<ValueType>::getData() const {
    return m_data;
}

template<class ValueType>
size_t Tensor<ValueType>::getLength() const {
    return length(m_dims);
}


template<class ValueType>
void Tensor<ValueType>::generateDimsSpan() {
    m_dimsSpan = genDimsSpan(m_dims);
}

template<class ValueType>
int Tensor<ValueType>::index2Offset(const vector<int> &index) const {
    return index2Offset(m_dimsSpan, index);
}

template<class ValueType>
int Tensor<ValueType>::index2Offset(const vector<int>& dimsSpan, const vector<int>& index) const{
    int N = index.size();
    int offset = 0;
    for (int i = 0; i < N; ++i) {
        offset += index[i] * dimsSpan[i];
    }
    return offset;
}

template<class ValueType>
vector<int> Tensor<ValueType>::offset2Index(const int offset) const {
    return offset2Index(m_dimsSpan, offset);
}

template<class ValueType>
vector<int> Tensor<ValueType>::offset2Index(const vector<int>& dimsSpan, const int offset) const{
    int dim = dimsSpan.size();
    vector<int> index(dim, 0);
    int n = offset;
    for (int i = 0; i < dim; ++i) {
        index[i] = n / dimsSpan[i];
        n -= index[i] * dimsSpan[i];
        if (0 == n) break;
    }
    assert(0 == n);
    return index;
}

template<class ValueType>
ValueType &Tensor<ValueType>::e(const vector<int> &index) const {
    assert(index.size() == m_dims.size());
    return m_data[index2Offset(index)];
}

template<class ValueType>
void Tensor<ValueType>::copyDataTo(Tensor *pTensor, const int lengthInValueType, const int srcOffsetInValueType) {
    memcpy(pTensor->m_data, m_data + srcOffsetInValueType, lengthInValueType * sizeof(ValueType));
}

template<class ValueType>
ValueType &Tensor<ValueType>::e(int index) const {
    return m_data[index];
}

template<class ValueType>
ValueType &Tensor<ValueType>::e(int i, int j) const {
    assert(2 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::e(int i, int j, int k) const {
    assert(3 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::e(int i, int j, int k, int l) const {
    assert(4 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2] + l * m_dimsSpan[3]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::e(int i, int j, int k, int l, int m) const {
    assert(5 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2] + l * m_dimsSpan[3] + m * m_dimsSpan[4]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::e(int i, int j, int k, int l, int m, int n) const {
    assert(6 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2] + l * m_dimsSpan[3] + m * m_dimsSpan[4] +
                  n * m_dimsSpan[5]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::e(int i, int j, int k, int l, int m, int n, int o) const {
    assert(7 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2] + l * m_dimsSpan[3] + m * m_dimsSpan[4] +
                  n * m_dimsSpan[5] + o * m_dimsSpan[6]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::operator[](int index) const {
    return m_data[index];
}

template<class ValueType>
ValueType &Tensor<ValueType>::operator()(int index) const {
    return m_data[index];
}

template<class ValueType>
ValueType &Tensor<ValueType>::operator()(int i, int j) const {
    assert(2 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::operator()(int i, int j, int k) const {
    assert(3 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::operator()(int i, int j, int k, int l) const {
    assert(4 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2] + l * m_dimsSpan[3]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::operator()(int i, int j, int k, int l, int m) const {
    assert(5 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2] + l * m_dimsSpan[3] + m * m_dimsSpan[4]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::operator()(int i, int j, int k, int l, int m, int n) const {
    assert(6 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2] + l * m_dimsSpan[3] + m * m_dimsSpan[4] +
                  n * m_dimsSpan[5]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::operator()(int i, int j, int k, int l, int m, int n, int o) const {
    assert(7 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2] + l * m_dimsSpan[3] + m * m_dimsSpan[4] +
                  n * m_dimsSpan[5] + o * m_dimsSpan[6]];
}

// transpose operation only supports 2D matrix
template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::transpose() {
    vector<int> newDims = reverseVector(m_dims);
    Tensor tensor(newDims);
    int dim = m_dims.size();
    assert(dim == 2);
    for (int i = 0; i < newDims[0]; ++i) {
        for (int j = 0; j < newDims[1]; ++j) {
            tensor.e({i, j}) = e({j, i});
        }
    }
    return tensor;
}


//Only support 2 dimensional tensor
template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator*(const Tensor<ValueType> &other) {
    int thisDim = m_dims.size();
    vector<int> otherDims = other.getDims();
    int otherDim = otherDims.size();
    if (m_dims[thisDim - 1] != otherDims[0]) {
        cout << "Error: Tensor product has un-matching dimension." << endl;
        return *this;
    }

    if (2 == thisDim && 2 == otherDim) {
        vector<int> newDims{m_dims[0], otherDims[1]};
        Tensor tensor(newDims);
        for (int i = 0; i < newDims[0]; ++i) {
            for (int j = 0; j < newDims[1]; ++j) {
                ValueType value = 0;
                for (int k = 0; k < m_dims[1]; ++k) {
                    value += e({i, k}) * other.e({k, j});
                }
                tensor.e({i, j}) = value;
            }
        }
        return tensor;
    } else {
        cout << "Error: Tensor product only support 2D tensor." << endl;
        return *this;
    }

}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator+(const float other) {
    Tensor tensor(m_dims);
    int N = tensor.getLength();
    for (int i = 0; i < N; ++i) {
        tensor.e(i) = e(i) + other;
    }
    return tensor;
}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator-(const float other) {
    Tensor tensor(m_dims);
    int N = tensor.getLength();
    for (int i = 0; i < N; ++i) {
        tensor.e(i) = e(i) - other;
    }
    return tensor;
}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator*(const float factor) {
    Tensor tensor(m_dims);
    int N = tensor.getLength();
    for (int i = 0; i < N; ++i) {
        tensor.e(i) = e(i) * factor;
    }
    return tensor;
}


template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator+(const Tensor<ValueType> &other) {
    assert(sameVector(m_dims, other.getDims()));
    Tensor tensor(m_dims);
    int N = getLength();
    for (int i = 0; i < N; ++i) {
        tensor.e(i) = e(i) + other.e(i);
    }
    return tensor;

}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator-(const Tensor<ValueType> &other) {
    assert(sameVector(m_dims, other.getDims()));
    Tensor tensor(m_dims);
    int N = getLength();
    for (int i = 0; i < N; ++i) {
        tensor.e(i) = e(i) - other.e(i);
    }
    return tensor;
}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator/(const float divisor) {
    if (0 == divisor) {
        return *this;
    }

    Tensor tensor(m_dims);
    int N = getLength();
    for (int i = 0; i < N; ++i) {
        tensor.e(i) = e(i) / divisor;
    }
    return tensor;
}

template<class ValueType>
Tensor<ValueType> &Tensor<ValueType>::operator+=(const Tensor &right) {
    assert(sameVector(m_dims, right.getDims()));
    int N = getLength();
    for (int i = 0; i < N; ++i) {
        e(i) += right.e(i);
    }
    return *this;
}

template<class ValueType>
Tensor<ValueType> &Tensor<ValueType>::operator-=(const Tensor &right) {
    assert(sameVector(m_dims, right.getDims()));
    int N = getLength();
    for (int i = 0; i < N; ++i) {
        e(i) -= right.e(i);
    }
    return *this;
}

template<class ValueType>
bool Tensor<ValueType>::operator==(const Tensor &right) {
    if (this == &right) return true;
    else {
        if (!sameVector(getDims(), right.getDims())) return false;
        else {
            int N = getLength();
            for (int i = 0; i < N; ++i) {
                if (e(i) != right.e(i)) return false;
            }
            return true;
        }
    }
}

template<class ValueType>
bool Tensor<ValueType>::operator!=(const Tensor &right) {
    return !(*this == right);
}

template<class ValueType>
Tensor<ValueType> &Tensor<ValueType>::operator+=(const float right) {
    int N = getLength();
    for (int i = 0; i < N; ++i) {
        e(i) += right;
    }
    return *this;
}

template<class ValueType>
Tensor<ValueType> &Tensor<ValueType>::operator-=(const float right) {
    int N = getLength();
    for (int i = 0; i < N; ++i) {
        e(i) -= right;
    }
    return *this;
}

template<class ValueType>
Tensor<ValueType> &Tensor<ValueType>::operator*=(const float factor) {
    int N = getLength();
    for (int i = 0; i < N; ++i) {
        e(i) *= factor;
    }
    return *this;
}

template<class ValueType>
Tensor<ValueType> &Tensor<ValueType>::operator/=(const float divisor) {
    if (0 != divisor) {
        int N = getLength();
        for (int i = 0; i < N; ++i) {
            e(i) /= divisor;
        }
    }
    return *this;
}



template<class ValueType>
float Tensor<ValueType>::sum() {
    int N = getLength();
    float sum = 0;
    //todo: sum use GPU will implement in the future, which will speed up from N to log(N)
    for (int i = 0; i < N; ++i) {
        sum += e(i);
    }
    return sum;
}

template<class ValueType>
float Tensor<ValueType>::average() {
    int N = getLength();
    if (0 == N) return 0;
    else {
        return sum() / N;
    }
}

template<class ValueType>
float Tensor<ValueType>::variance() {
    float mu = average();
    int N = getLength();
    if (1 == N || 0 == N) return 0;
    float sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += pow((e(i) - mu), 2);
    }
    return sum / (N - 1); // population variance
}

template<class ValueType>
float Tensor<ValueType>::max() {
    int N = getLength();
    float maxValue = (float) e(0);
    for (int i = 1; i < N; ++i) {
        if (e(i) > maxValue) maxValue = e(i);
    }
    return maxValue;
}

template<class ValueType>
float Tensor<ValueType>::min() {
    int N = getLength();
    float minValue = (float) e(0);
    for (int i = 1; i < N; ++i) {
        if (e(i) < minValue) minValue = (float) e(i);
    }
    return minValue;
}

template<class ValueType>
void Tensor<ValueType>::getMinMax(ValueType& min, ValueType& max){
    const int N = getLength();
    min = e(0);
    max = e(0);
    for (int i = 1; i < N; ++i) {
        if (e(i) < min) min = e(i);
        if (e(i) > max) max = e(i);
    }
}

template<class ValueType>
int Tensor<ValueType>::maxPosition() {
    int N = getLength();
    ValueType maxValue = e(0);
    int maxPos = 0;
    for (int i = 1; i < N; ++i) {
        if (e(i) > maxValue) {
            maxValue = e(i);
            maxPos = i;
        }
    }
    return maxPos;
}

/* getMaxPositionSubTensor():
 * gets all indexes of max value along the 0th dimension, then combine them into a subTensor.
 *
 * */
template<class ValueType>
Tensor<unsigned  char> Tensor<ValueType>::getMaxPositionSubTensor() {
    vector<int> subTensorDims = m_dims;
    subTensorDims.erase(subTensorDims.begin());
    Tensor<unsigned  char> subTensor(subTensorDims);
    int compareN = m_dims[0];
    int N = subTensor.getLength();
    for (int j = 0; j < N; ++j) {
        int maxIndex = 0;
        ValueType maxValue = e(j);
        for (int i = 1; i < compareN; ++i) {
            if (e(i * N + j) > maxValue) {
                maxValue = e(i * N + j);
                maxIndex = i;
            }
        }
        subTensor(j) = (unsigned  char) maxIndex;
    }
    return subTensor;
}

template<class ValueType>
float Tensor<ValueType>::L2Norm(){
    const int N = getLength();
    float sum = 0.0;
    for (int i=0; i<N; ++i){
        sum += e(i)*e(i);
    }
    return sqrt(sum);
}

template<class ValueType>
void Tensor<ValueType>::save(const string& fullFilename, bool matrix2D){
    FILE * pFile = nullptr;
    pFile = fopen (fullFilename.c_str(),"w");
    if (nullptr == pFile){
        printf("Error: can not open  %s  file for writing.\n", fullFilename.c_str());
        return;
    }
    if (matrix2D && 2 == m_dims.size()){
        for (int i= 0; i< m_dims[0]; ++i){
            for (int j=0; j<m_dims[1]; ++j){
                fprintf(pFile, "%f ", e(i,j));
            }
            fprintf(pFile, "\r\n");
        }
        fprintf(pFile, "\r\n");
    }
    else {
        int N = getLength();
        for (int i = 0; i < N; ++i) {
            fprintf(pFile, "%f ", e(i));
        }
        fprintf(pFile, "\r\n");
    }
    fclose (pFile);
}

template<class ValueType>
void Tensor<ValueType>::print(bool fixWidth){
    if (2 == m_dims.size()){
        for (int i= 0; i< m_dims[0]; ++i){
            for (int j=0; j<m_dims[1]; ++j){
                if (fixWidth){
                    printf("%3d", int(e(i,j)));
                }else{
                    printf("%f ", (float)e(i,j));
                }
            }
            printf("\n");
        }
        printf("\n");
    }
    else {
        int N = getLength();
        for (int i = 0; i < N; ++i) {
            printf("%f ", (float) e(i));
        }
        printf("\n");
    }
}

template<class ValueType>
bool Tensor<ValueType>::load(const string& fullFilename, bool matrix2D){
    FILE * pFile = nullptr;
    pFile = fopen (fullFilename.c_str(),"r");
    if (nullptr == pFile){
        printf("Error: can not open  %s  file for reading.\n", fullFilename.c_str());
        return false;
    }
    if (matrix2D && 2 == m_dims.size()){
        for (int i= 0; i< m_dims[0]; ++i){
            for (int j=0; j<m_dims[1]; ++j){
                int count = fscanf(pFile, "%f ", &e(i,j));
                if (1 != count){
                    printf("Error: fscanf in Tensor load method.\n");
                }
            }
            //int count = fscanf(pFile, "%*c");

        }
    }
    else {
        int N = getLength();
        for (int i = 0; i < N; ++i) {
            int count = fscanf(pFile, "%f ", &e(i));
            if (1 != count){
                printf("Error: fscanf in Tensor load method.\n");
            }
        }
    }
    fclose (pFile);
    return true;
}

//natural logarithm
template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::ln() {
    Tensor tensor(m_dims);
    int N = getLength();
    for (int i = 0; i < N; ++i) {
        tensor.e(i) = log(e(i));
    }
    return tensor;
}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::expon(){
    Tensor tensor(m_dims);
    int N = getLength();
    for (int i = 0; i < N; ++i) {
        tensor.e(i) = exp(e(i));
    }
    return tensor;

}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::sign(){
    Tensor tensor(m_dims);
    int N = getLength();
    for (int i = 0; i < N; ++i) {
        tensor.e(i) = e(i)>= 0 ? (e(i)>0? 1:0):-1;
    }
    return tensor;
}

//element-wise product

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::hadamard(const Tensor &right) {
    assert(sameVector(m_dims, right.m_dims));
    Tensor tensor(m_dims);
    int N = getLength();
    for (int i = 0; i < N; ++i) {
        tensor.e(i) = e(i) * right.e(i);
    }
    return tensor;
}

template<class ValueType>
float Tensor<ValueType>::dotProduct(const Tensor &right) {
    assert(sameVector(m_dims, right.m_dims));
    Tensor hadamardTensor = this->hadamard(right);
    return hadamardTensor.sum();
}


template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::vectorize() {
    Tensor tensor = *this;
    tensor.m_dims = {getLength(), 1};
    tensor.generateDimsSpan();
    return tensor;
}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::reshape(vector<int> newDims) {
    int dim = newDims.size();
    int newN = 1;
    for (int i = 0; i < dim; ++i) {
        newN *= newDims[i];
    }
    if (newN != getLength()) {
        cout << "Error: Tensor reshape has different length. " << endl;
        return *this;
    } else {
        Tensor tensor = *this;
        tensor.m_dims = newDims;
        tensor.generateDimsSpan();
        return tensor;
    }
}

template<class ValueType>
void Tensor<ValueType>::subTensorFromTopLeft(const vector<int> &tlIndex, Tensor *pTensor, const vector<int>& stride) const {
    assert(pTensor->getDims().size() == tlIndex.size());
    subTensorFromTopLeft(index2Offset(tlIndex), pTensor, stride);
}

template<class ValueType>
void Tensor<ValueType>::subTensorFromTopLeft(const int  offset, Tensor* pTensor, const vector<int>& stride) const {
    vector<int> dims = pTensor->getDims();
    const int dim = dims.size();
    const size_t  elementLen = sizeof(ValueType);
    const size_t  rowLen = dims[dim-1]*elementLen;

    ValueType* dstOffset = pTensor->m_data;
    ValueType* srcOffset = m_data+offset;
    vector<int> index(dim, 0);
    int p = dim-2;
    while (index[0] != dims[0]) {
        int dstNum = 0;
        int srcNum = 0;
        for (int i=0; i< dim-1; ++i){
            dstNum += pTensor->m_dimsSpan[i]*index[i];
            srcNum += m_dimsSpan[i]*index[i]*stride[i];
        }
        if (isElementEqual1(stride)){
            memcpy(dstOffset+dstNum, srcOffset+srcNum, rowLen);
        }
        else{
            for (int i=0; i< dims[dim-1]; ++i){
                memcpy(dstOffset+dstNum+i, srcOffset+srcNum+i*stride[dim-1], elementLen);
            }
        }

        index[p]++;
        while(index[p]==dims[p] && p >0) {
            index[p]=0;
            index[--p]++;
            if (index[p] != dims[p]) {
                p=dim-2;
                break;
            }
        }
    }
}

template<class ValueType>
void Tensor<ValueType>::dilute(Tensor* & pTensor, const vector<int>& tensorSizeBeforeCollapse, const vector<int>& filterSize, const vector<int>& stride) const{
    assert(tensorSizeBeforeCollapse.size() == filterSize.size());
    assert(length(tensorSizeBeforeCollapse) == length(m_dims));
    const int dim = tensorSizeBeforeCollapse.size();
    vector<int> newTensorSize(dim, 0);
    for (int i=0; i< dim; ++i){
        newTensorSize[i] = tensorSizeBeforeCollapse[i]* stride[i] + filterSize[i]*2 -1;
        /*analysis:
        Convolution forward: O = (I-F)/S + 1;
         O*S -S = I-F
         (O-1)*S +F = I
         considering that (I-F)/S may loss remainder less than S, so the (O-1)*S +F<= real I < O*S +F.
         then considering padding of F-1 in both side of one dimension.
         therefore, I = O*S +F*2-1
        */
    }
    pTensor = new Tensor<ValueType>(newTensorSize);
    pTensor->zeroInitialize();
    const int N = getLength();
    vector<int> dimsSpanBeforeCollapse = genDimsSpan(tensorSizeBeforeCollapse);
    for (int oldOffset=0; oldOffset<N; ++oldOffset){
        vector<int> oldIndex = offset2Index(dimsSpanBeforeCollapse, oldOffset);
        vector<int> newIndex = oldIndex* stride+ filterSize -1;
        int newOffset = pTensor->index2Offset(newIndex);
        pTensor->e(newOffset) = e(oldOffset);
    }
}

template<class ValueType>
void Tensor<ValueType>::putInBiggerTensor(Tensor* pBiggerTensor, const vector<int>& offsetVec, const vector<int>& stride) const {
   assert(m_dims.size() == pBiggerTensor->getDims().size());
   const int N = getLength();
   for (int smallOffset=0; smallOffset<N; ++smallOffset){
        vector<int> smallIndex = offset2Index(smallOffset);
        vector<int> bigIndex = smallIndex* stride+ offsetVec;
        const int bigOffset = pBiggerTensor->index2Offset(bigIndex);
        pBiggerTensor->e(bigOffset) = e(smallOffset);
    }
}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::column(const int index) {
    assert(2 == m_dims.size());
    vector<int> newDims;
    newDims.push_back(m_dims[0]);
    newDims.push_back(1);
    Tensor<ValueType> tensor(newDims);
    for (int i = 0; i < m_dims[0]; ++i) {
        tensor.e(i) = e(i, index);
    }
    return tensor;
}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::row(const int index) {
    assert(2 == m_dims.size());
    vector<int> newDims;
    newDims.push_back(1);
    newDims.push_back(m_dims[1]);
    Tensor<ValueType> tensor(newDims);
    copyDataTo(&tensor, m_dims[1], index * m_dims[1]);
    return tensor;
}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::slice(const int index) {
    assert(3 == m_dims.size());
    vector<int> newDims;
    newDims = m_dims;
    newDims.erase(newDims.begin());
    Tensor<ValueType> tensor(newDims);
    int N = tensor.getLength();
    copyDataTo(&tensor, N,  index * N);
    return tensor;
}

template<class ValueType>
void Tensor<ValueType>::volume(const int index, Tensor *&pTensor) {
    assert(4 == m_dims.size());
    vector<int> newDims;
    newDims = m_dims;
    newDims.erase(newDims.begin());
    pTensor = new Tensor<ValueType>(newDims);
    int N = pTensor->getLength();
    copyDataTo(pTensor, N, index * N);

}

template<class ValueType>
void Tensor<ValueType>::fourDVolume(const int index, Tensor *&pTensor) {
    assert(5 == m_dims.size());
    vector<int> newDims;
    newDims = m_dims;
    newDims.erase(newDims.begin());
    pTensor = new Tensor<ValueType>(newDims);
    int N = pTensor->getLength();
    copyDataTo(pTensor, N, index * N);
}

template<class ValueType>
void Tensor<ValueType>::extractLowerDTensor(const int index, Tensor *&pTensor) {
    vector<int> newDims;
    newDims = m_dims;
    newDims.erase(newDims.begin());
    if (1 == newDims.size()) {
        newDims.insert(newDims.begin(), 1);
    }
    pTensor = new Tensor<ValueType>(newDims);
    int N = pTensor->getLength();
    copyDataTo(pTensor, N, index * N);
}


//convolution or cross-correlation
template<class ValueType>
float Tensor<ValueType>::conv(const Tensor &right, int nThreads) const {
    assert(sameLength(m_dims, right.getDims()));
    const int N = getLength();
    nThreads = (N < 1000)? 1 : nThreads;
    float sum = 0.0;
    if (1 == nThreads) {
        for (int i = 0; i < N; ++i) {
            sum += e(i) * right.e(i);
        }
    }
    else {
        const int NRange = (N + nThreads -1)/nThreads;
        float *partSum = new float[nThreads];
        vector<std::thread> threadVec;
        for (int t = 0; t < nThreads; ++t) {
            threadVec.push_back(thread([this, N, partSum, t, &right, NRange]() {
                                           partSum[t] = 0;
                                           for (int i = NRange * t; i < NRange * (t + 1) && i < N; ++i) {
                                               partSum[t] += e(i) * right.e(i);
                                           }
                                       }
            ));
        }
        for (int t = 0; t < threadVec.size(); ++t) {
            threadVec[t].join();
        }
        for (int i=0; i< nThreads; ++i){
            sum += partSum[i];
        }
        delete[] partSum;
    }
    return sum;
}

template<class ValueType>
float Tensor<ValueType>::flipConv(const Tensor &right, int nThreads) const {
    assert(sameLength(m_dims, right.getDims()));
    int N = getLength();
    nThreads = (N < 1000)? 1 : nThreads;
    float sum = 0.0;
    if (1 == nThreads) {
        for (int i = 0; i < N; ++i) {
            sum += e(N - i - 1) * right.e(i);
        }
    }
    else {
        const int NRange = (N + nThreads -1)/nThreads;
        float *partSum = new float[nThreads];
        vector<std::thread> threadVec;
        for (int t = 0; t < nThreads; ++t) {
            threadVec.push_back(thread([this, N, partSum, t, &right, NRange]() {
                                           partSum[t] = 0;
                                           for (int i = NRange * t; i < NRange * (t + 1) && i < N; ++i) {
                                               partSum[t] += e(N-i-1) * right.e(i);
                                           }
                                       }
            ));
        }
        for (int t = 0; t < threadVec.size(); ++t) {
            threadVec[t].join();
        }
        for (int i=0; i< nThreads; ++i){
            sum += partSum[i];
        }
        delete[] partSum;
    }
    return sum;
}

template<class ValueType>
Tensor<ValueType> &Tensor<ValueType>::flip() {
    int N = getLength();
    int M = N / 2;
    ValueType temp = 0;
    for (int i = 0; i < M; ++i) {
        temp = e(i);
        e(i) = e(N - 1 - i);
        e(N - 1 - i) = temp;
    }
    return *this;
}

template<class ValueType>
vector<int> Tensor<ValueType>::getCenterOfNonZeroElements() {
    const int size = getDims().size();
    vector<int> center(size,0 );
    vector<int> sumPos(size, 0);
    const int N = getLength();
    int nCount = 0;
    for(int i=0; i< N; ++i){
        if (e(i)>0){
            vector<int> index = offset2Index(i);
            sumPos = sumPos + index;
            ++nCount;
        }
    }

    if (0 != nCount){
        center = sumPos/nCount;
    }

    if (center != vector<int>(size, 0)){
        return center;
    }
    else{
        cout<<"Error: can not find center of NonZeroelements"<<endl;
        std::exit(EXIT_FAILURE);
    }
}

template<class ValueType>
void
Tensor<ValueType>::rotate3D(const vector<float> radianVec, const int interpolation, Tensor<float> *&pRotatedTensor) {

    if (3 != m_dims.size() || 3 != radianVec.size()){
        cout<<"Error: rotate3D is only for 3D Tensor. Exit."<<endl;
        return;
    }

    const Ipp32f* pSrc = m_data;
    IpprVolume srcVolume;
    srcVolume.width = m_dims[2]; srcVolume.height = m_dims[1];  srcVolume.depth = m_dims[0];
    //srcVolume.width = m_dims[1]; srcVolume.height = m_dims[0];  srcVolume.depth = m_dims[2];
    int srcStep = sizeof(ValueType)* srcVolume.width;
    int srcPlaneStep = sizeof(ValueType)* srcVolume.width* srcVolume.height;

    IpprCuboid srcVoi; // Volume of Interest
    srcVoi.x = 0; srcVoi.y = 0; srcVoi.z = 0;
    srcVoi.width = srcVolume.width; srcVoi.height = srcVolume.height; srcVoi.depth = srcVolume.depth;

    double c[3][4] = {0}; //Ratation Matrix, all element initialize to zeros.
    getRotationMatrix(radianVec, c);

    const vector<int> dimRange = getMaxRangeOfHingeRotation(c, m_dims);
    IpprCuboid dstVoi;
    dstVoi.x = 0; dstVoi.y = 0;dstVoi.z = 0;
    dstVoi.width  = dimRange[2];
    dstVoi.height = dimRange[1];
    dstVoi.depth = dimRange[0];

    pRotatedTensor = new Tensor<float>(dimRange);
    Ipp32f* pDst = pRotatedTensor->m_data;
    int dstStep = sizeof(float)* dstVoi.width;
    int dstPlaneStep = sizeof(float)* dstVoi.width* dstVoi.height;

    checkIPP(ippInit());

    const int nChannel  =1 ;
    int bufferSize = 0;
    checkIPP(ipprWarpAffineGetBufSize(srcVolume, srcVoi, dstVoi, c, nChannel, interpolation, &bufferSize));
    Ipp8u* pBuffer = (Ipp8u*)ippMalloc(bufferSize);

    checkIPP(ipprWarpAffine_32f_C1V(pSrc, srcVolume, srcStep, srcPlaneStep, srcVoi,
                                    pDst, dstStep, dstPlaneStep,dstVoi,
                                    c, interpolation, pBuffer));

    ippFree(pBuffer);

}

template<class ValueType>
void Tensor<ValueType>::rotate3D_NearestNeighbor(const vector<float> radianVec, Tensor<float> *&pRotatedTensor) {

    if (3 != m_dims.size() || 3 != radianVec.size()){
        cout<<"Error: rotate3D_naive is only for 3D Tensor. Exit."<<endl;
        return;
    }
    double c[3][4] = {0}; //Ratation Matrix, all element initialize to zeros.
    getRotationMatrix(radianVec, c);

    const vector<int> dimRange = getMaxRangeOfHingeRotation(c, m_dims);
    pRotatedTensor = new Tensor<float>(dimRange);
    pRotatedTensor->zeroInitialize();
    int n =0;
    for (int x= 0; x<m_dims[0]; ++x)
        for (int y=0; y<m_dims[1]; ++y)
            for (int z=0; z<m_dims[2]; ++z){
                int x1 = abs(c[0][0]*x+ c[0][1]*y + c[0][2]*z) +0.5;
                int y1 = abs(c[1][0]*x+ c[1][1]*y + c[1][2]*z) +0.5;
                int z1 = abs(c[2][0]*x+ c[2][1]*y + c[2][2]*z) +0.5;

                pRotatedTensor->e({x1,y1,z1}) = e(n++);
            }
}

