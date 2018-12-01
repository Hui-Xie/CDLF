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

#ifdef Use_GPU
  #include <cuda_runtime.h>
  #include "TensorCuda.h"
  #include "GPUAttr.h"
#endif



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
Tensor<ValueType>::Tensor(const vector<long> &dims) {
    initializeMember();
    m_dims = dims;
    if (1 == m_dims.size()) {
        m_dims.push_back(1);
    }
    generateDimsSpan();
    allocateMem();
}

template<class ValueType>
void Tensor<ValueType>::setDimsAndAllocateMem(const vector<long>& dims){
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
    long N = getLength();
    for (long i = 0; i < N; ++i) {
        e(i) = 0;
    }
}

template<class ValueType>
void Tensor<ValueType>::uniformInitialize(const ValueType x) {
    long N = getLength();
    for (long i = 0; i < N; ++i) {
        e(i) = x;
    }
}


template<class ValueType>
Tensor<ValueType> &Tensor<ValueType>::operator=(const Tensor &other) {
    if (this != &other) {
        freeMem();
        m_dims = other.getDims();
        generateDimsSpan();
        allocateMem();
        long length = other.getLength();
        if (length > 0) {
        memcpy(m_data, other.getData(), length * sizeof(ValueType));
        }
    }
    return *this;
}

template<class ValueType>
Tensor<ValueType>::~Tensor() {
    freeMem();
}

template<class ValueType>
void Tensor<ValueType>::copyDataFrom(void *buff, const long numBytes) {
    if (numBytes > getLength() * sizeof(ValueType)) {
        cout << "Error: numBytes of Tensor::copyDataFrom is bigger than data space." << endl;
        return;
    } else {
        memcpy(m_data, buff, numBytes);
    }
}

template<class ValueType>
vector<long> Tensor<ValueType>::getDims() const {
    return m_dims;
}

template<class ValueType>
vector<long> Tensor<ValueType>::getDimsSpan() const{
    return m_dimsSpan;
}

template<class ValueType>
ValueType *Tensor<ValueType>::getData() const {
    return m_data;
}

template<class ValueType>
long Tensor<ValueType>::getLength() const {
    return length(m_dims);
}


template<class ValueType>
void Tensor<ValueType>::generateDimsSpan() {
    m_dimsSpan = genDimsSpan(m_dims);
}

template<class ValueType>
long Tensor<ValueType>::index2Offset(const vector<long> &index) const {
    return index2Offset(m_dimsSpan, index);
}

template<class ValueType>
long Tensor<ValueType>::index2Offset(const vector<long>& dimsSpan, const vector<long>& index) const{
    int N = index.size();
    long offset = 0;
    for (int i = 0; i < N; ++i) {
        offset += index[i] * dimsSpan[i];
    }
    return offset;
}

template<class ValueType>
vector<long> Tensor<ValueType>::offset2Index(const long offset) const {
    return offset2Index(m_dimsSpan, offset);
}

template<class ValueType>
vector<long> Tensor<ValueType>::offset2Index(const vector<long>& dimsSpan, const long offset) const{
    int dim = dimsSpan.size();
    vector<long> index(dim, 0);
    long n = offset;
    for (int i = 0; i < dim; ++i) {
        index[i] = n / dimsSpan[i];
        n -= index[i] * dimsSpan[i];
        if (0 == n) break;
    }
    assert(0 == n);
    return index;
}

template<class ValueType>
ValueType &Tensor<ValueType>::e(const vector<long> &index) const {
    assert(index.size() == m_dims.size());
    return m_data[index2Offset(index)];
}

template<class ValueType>
void Tensor<ValueType>::copyDataTo(Tensor *pTensor, const long offset, const long length) {
    memcpy(pTensor->m_data, m_data + offset, length * sizeof(ValueType));
}

template<class ValueType>
ValueType &Tensor<ValueType>::e(long index) const {
    return m_data[index];
}

template<class ValueType>
ValueType &Tensor<ValueType>::e(long i, long j) const {
    assert(2 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::e(long i, long j, long k) const {
    assert(3 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::e(long i, long j, long k, long l) const {
    assert(4 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2] + l * m_dimsSpan[3]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::e(long i, long j, long k, long l, long m) const {
    assert(5 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2] + l * m_dimsSpan[3] + m * m_dimsSpan[4]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::e(long i, long j, long k, long l, long m, long n) const {
    assert(6 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2] + l * m_dimsSpan[3] + m * m_dimsSpan[4] +
                  n * m_dimsSpan[5]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::e(long i, long j, long k, long l, long m, long n, long o) const {
    assert(7 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2] + l * m_dimsSpan[3] + m * m_dimsSpan[4] +
                  n * m_dimsSpan[5] + o * m_dimsSpan[6]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::operator[](long index) const {
    return m_data[index];
}

template<class ValueType>
ValueType &Tensor<ValueType>::operator()(long index) const {
    return m_data[index];
}

template<class ValueType>
ValueType &Tensor<ValueType>::operator()(long i, long j) const {
    assert(2 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::operator()(long i, long j, long k) const {
    assert(3 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::operator()(long i, long j, long k, long l) const {
    assert(4 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2] + l * m_dimsSpan[3]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::operator()(long i, long j, long k, long l, long m) const {
    assert(5 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2] + l * m_dimsSpan[3] + m * m_dimsSpan[4]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::operator()(long i, long j, long k, long l, long m, long n) const {
    assert(6 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2] + l * m_dimsSpan[3] + m * m_dimsSpan[4] +
                  n * m_dimsSpan[5]];
}

template<class ValueType>
ValueType &Tensor<ValueType>::operator()(long i, long j, long k, long l, long m, long n, long o) const {
    assert(7 == m_dims.size());
    return m_data[i * m_dimsSpan[0] + j * m_dimsSpan[1] + k * m_dimsSpan[2] + l * m_dimsSpan[3] + m * m_dimsSpan[4] +
                  n * m_dimsSpan[5] + o * m_dimsSpan[6]];
}

// transpose operation only supports 2D matrix
template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::transpose() {
    vector<long> newDims = reverseVector(m_dims);
    Tensor tensor(newDims);
    int dim = m_dims.size();
    assert(dim == 2);
    for (long i = 0; i < newDims[0]; ++i) {
        for (long j = 0; j < newDims[1]; ++j) {
            tensor.e({i, j}) = e({j, i});
        }
    }
    return tensor;
}


//Only support 2 dimensional tensor
template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator*(const Tensor<ValueType> &other) {
    int thisDim = m_dims.size();
    vector<long> otherDims = other.getDims();
    int otherDim = otherDims.size();
    if (m_dims[thisDim - 1] != otherDims[0]) {
        cout << "Error: Tensor product has un-matching dimension." << endl;
        return *this;
    }

    if (2 == thisDim && 2 == otherDim) {
        vector<long> newDims{m_dims[0], otherDims[1]};
        Tensor tensor(newDims);
        for (long i = 0; i < newDims[0]; ++i) {
            for (long j = 0; j < newDims[1]; ++j) {
                ValueType value = 0;
                for (long k = 0; k < m_dims[1]; ++k) {
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
    long N = tensor.getLength();
    for (long i = 0; i < N; ++i) {
        tensor.e(i) = e(i) + other;
    }
    return tensor;
}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator-(const float other) {
    Tensor tensor(m_dims);
    long N = tensor.getLength();
    for (long i = 0; i < N; ++i) {
        tensor.e(i) = e(i) - other;
    }
    return tensor;
}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator*(const float factor) {
    Tensor tensor(m_dims);
    long N = tensor.getLength();
    for (long i = 0; i < N; ++i) {
        tensor.e(i) = e(i) * factor;
    }
    return tensor;
}


template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator+(const Tensor<ValueType> &other) {
    assert(sameVector(m_dims, other.getDims()));
    Tensor tensor(m_dims);
    long N = getLength();
    for (long i = 0; i < N; ++i) {
        tensor.e(i) = e(i) + other.e(i);
    }
    return tensor;

}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::operator-(const Tensor<ValueType> &other) {
    assert(sameVector(m_dims, other.getDims()));
    Tensor tensor(m_dims);
    long N = getLength();
    for (long i = 0; i < N; ++i) {
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
    long N = getLength();
    for (long i = 0; i < N; ++i) {
        tensor.e(i) = e(i) / divisor;
    }
    return tensor;
}

template<class ValueType>
Tensor<ValueType> &Tensor<ValueType>::operator+=(const Tensor &right) {
    assert(sameVector(m_dims, right.getDims()));
    long N = getLength();
    for (long i = 0; i < N; ++i) {
        e(i) += right.e(i);
    }
    return *this;
}

template<class ValueType>
Tensor<ValueType> &Tensor<ValueType>::operator-=(const Tensor &right) {
    assert(sameVector(m_dims, right.getDims()));
    long N = getLength();
    for (long i = 0; i < N; ++i) {
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
            long N = getLength();
            for (long i = 0; i < N; ++i) {
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
    long N = getLength();
    for (long i = 0; i < N; ++i) {
        e(i) += right;
    }
    return *this;
}

template<class ValueType>
Tensor<ValueType> &Tensor<ValueType>::operator-=(const float right) {
    long N = getLength();
    for (long i = 0; i < N; ++i) {
        e(i) -= right;
    }
    return *this;
}

template<class ValueType>
Tensor<ValueType> &Tensor<ValueType>::operator*=(const float factor) {
    long N = getLength();
    for (long i = 0; i < N; ++i) {
        e(i) *= factor;
    }
    return *this;
}

template<class ValueType>
Tensor<ValueType> &Tensor<ValueType>::operator/=(const float divisor) {
    if (0 != divisor) {
        long N = getLength();
        for (long i = 0; i < N; ++i) {
            e(i) /= divisor;
        }
    }
    return *this;
}



template<class ValueType>
float Tensor<ValueType>::sum() {
    long N = getLength();
    float sum = 0;
    //todo: sum use GPU will implement in the future, which will speed up from N to log(N)
    for (long i = 0; i < N; ++i) {
        sum += e(i);
    }
    return sum;
}

template<class ValueType>
float Tensor<ValueType>::average() {
    long N = getLength();
    if (0 == N) return 0;
    else {
        return sum() / N;
    }
}

template<class ValueType>
float Tensor<ValueType>::variance() {
    float mu = average();
    long N = getLength();
    if (1 == N || 0 == N) return 0;
    float sum = 0;
    for (long i = 0; i < N; ++i) {
        sum += pow((e(i) - mu), 2);
    }
    return sum / (N - 1); // population variance
}

template<class ValueType>
float Tensor<ValueType>::max() {
    long N = getLength();
    float maxValue = (float) e(0);
    for (long i = 1; i < N; ++i) {
        if (e(i) > maxValue) maxValue = e(i);
    }
    return maxValue;
}

template<class ValueType>
float Tensor<ValueType>::min() {
    long N = getLength();
    float minValue = (float) e(0);
    for (long i = 1; i < N; ++i) {
        if (e(i) < minValue) minValue = (float) e(i);
    }
    return minValue;
}

template<class ValueType>
long Tensor<ValueType>::maxPosition() {
    long N = getLength();
    ValueType maxValue = e(0);
    long maxPos = 0;
    for (long i = 1; i < N; ++i) {
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
Tensor<unsigned char> Tensor<ValueType>::getMaxPositionSubTensor() {
    vector<long> subTensorDims = m_dims;
    subTensorDims.erase(subTensorDims.begin());
    Tensor<unsigned char> subTensor(subTensorDims);
    int compareN = m_dims[0];
    long N = subTensor.getLength();
    for (long j = 0; j < N; ++j) {
        int maxIndex = 0;
        int maxValue = e(j);
        for (int i = 1; i < compareN; ++i) {
            if (e(i * N + j) > maxValue) {
                maxValue = e(i * N + j);
                maxIndex = i;
            }
        }
        subTensor(j) = (unsigned char) maxIndex;
    }
    return subTensor;
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
        long N = getLength();
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
                    printf("%f ", e(i,j));
                }
            }
            printf("\n");
        }
        printf("\n");
    }
    else {
        long N = getLength();
        for (int i = 0; i < N; ++i) {
            printf("%f ", e(i));
        }
        printf("\n");
    }
}

template<class ValueType>
void Tensor<ValueType>::load(const string& fullFilename, bool matrix2D){
    FILE * pFile = nullptr;
    pFile = fopen (fullFilename.c_str(),"r");
    if (nullptr == pFile){
        printf("Error: can not open  %s  file for reading.\n", fullFilename.c_str());
        return;
    }
    if (matrix2D && 2 == m_dims.size()){
        for (int i= 0; i< m_dims[0]; ++i){
            for (int j=0; j<m_dims[1]; ++j){
                fscanf(pFile, "%f ", &e(i,j));
            }
            fscanf(pFile, "%*c");
        }
    }
    else {
        long N = getLength();
        for (int i = 0; i < N; ++i) {
            fscanf(pFile, "%f ", &e(i));
        }
    }
    fclose (pFile);
}

//natural logarithm
template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::ln() {
    Tensor tensor(m_dims);
    long N = getLength();
    for (long i = 0; i < N; ++i) {
        tensor.e(i) = log(e(i));
    }
    return tensor;
}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::expon(){
    Tensor tensor(m_dims);
    long N = getLength();
    for (long i = 0; i < N; ++i) {
        tensor.e(i) = exp(e(i));
    }
    return tensor;

}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::sign(){
    Tensor tensor(m_dims);
    long N = getLength();
    for (long i = 0; i < N; ++i) {
        tensor.e(i) = e(i)>= 0 ? (e(i)>0? 1:0):-1;
    }
    return tensor;
}

//element-wise product

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::hadamard(const Tensor &right) {
    assert(sameVector(m_dims, right.m_dims));
    Tensor tensor(m_dims);
    long N = getLength();
    for (long i = 0; i < N; ++i) {
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
Tensor<ValueType> Tensor<ValueType>::reshape(vector<long> newDims) {
    int dim = newDims.size();
    long newN = 1;
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
void Tensor<ValueType>::subTensorFromTopLeft(const vector<long> &tlIndex, Tensor *pTensor, const int stride) const {
    assert(pTensor->getDims().size() == tlIndex.size());
    subTensorFromTopLeft(index2Offset(tlIndex), pTensor, stride);
}

template<class ValueType>
void Tensor<ValueType>::subTensorFromTopLeft(const long  offset, Tensor* pTensor, const int stride) const {
    vector<long> dims = pTensor->getDims();
    const int dim = dims.size();
    size_t  elementLen = sizeof(ValueType);
    size_t  rowLen = dims[dim-1]*elementLen;

    ValueType* dstOffset = pTensor->m_data;
    ValueType* srcOffset = m_data+offset;
    vector<long> index(dim, 0);
    int p = dim-2;
    while (index[0] != dims[0]) {
        long dstNum = 0;
        long srcNum = 0;
        for (int i=0; i< dim-1; ++i){
            dstNum += pTensor->m_dimsSpan[i]*index[i];
            srcNum += m_dimsSpan[i]*index[i]*stride;
        }
        if (1 == stride){
            memcpy(dstOffset+dstNum, srcOffset+srcNum, rowLen);
        }
        else{
            for (int i=0; i< dims[dim-1]; ++i){
                memcpy(dstOffset+dstNum+i, srcOffset+srcNum+i*stride, elementLen);
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
void Tensor<ValueType>::dilute(Tensor* & pTensor, const vector<long>& tensorSizeBeforeCollapse, const vector<long>& paddingWidthVec, const int stride) const{
    assert(tensorSizeBeforeCollapse.size() == paddingWidthVec.size());
    assert(length(tensorSizeBeforeCollapse) == length(m_dims));
    const int dim = tensorSizeBeforeCollapse.size();
    vector<long> newTensorSize(dim, 0);
    for (int i=0; i< dim; ++i){
        newTensorSize[i] = (tensorSizeBeforeCollapse[i]-1)* stride + 2 + paddingWidthVec[i]*2;
        // in above, "+2 " is to make sure the inputSize =even still get correct diluted Tensor
    }
    pTensor = new Tensor<ValueType>(newTensorSize);
    pTensor->zeroInitialize();
    long N = getLength();
    vector<long> dimsSpanBeforeCollapse = genDimsSpan(tensorSizeBeforeCollapse);
    for (long oldOffset=0; oldOffset<N; ++oldOffset){
        vector<long> oldIndex = offset2Index(dimsSpanBeforeCollapse, oldOffset);
        vector<long> newIndex = oldIndex* stride+ paddingWidthVec;
        long newOffset = pTensor->index2Offset(newIndex);
        pTensor->e(newOffset) = e(oldOffset);
    }
}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::column(const int index) {
    assert(2 == m_dims.size());
    vector<long> newDims;
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
    vector<long> newDims;
    newDims.push_back(1);
    newDims.push_back(m_dims[1]);
    Tensor<ValueType> tensor(newDims);
    copyDataTo(&tensor, index * m_dims[1], m_dims[1]);
    return tensor;
}

template<class ValueType>
Tensor<ValueType> Tensor<ValueType>::slice(const int index) {
    assert(3 == m_dims.size());
    vector<long> newDims;
    newDims = m_dims;
    newDims.erase(newDims.begin());
    Tensor<ValueType> tensor(newDims);
    long N = tensor.getLength();
    copyDataTo(&tensor, index * N, N);
    return tensor;
}

template<class ValueType>
void Tensor<ValueType>::volume(const int index, Tensor *&pTensor) {
    assert(4 == m_dims.size());
    vector<long> newDims;
    newDims = m_dims;
    newDims.erase(newDims.begin());
    pTensor = new Tensor<ValueType>(newDims);
    long N = pTensor->getLength();
    copyDataTo(pTensor, index * N, N);

}

template<class ValueType>
void Tensor<ValueType>::fourDVolume(const int index, Tensor *&pTensor) {
    assert(5 == m_dims.size());
    vector<long> newDims;
    newDims = m_dims;
    newDims.erase(newDims.begin());
    pTensor = new Tensor<ValueType>(newDims);
    long N = pTensor->getLength();
    copyDataTo(pTensor, index * N, N);
}

template<class ValueType>
void Tensor<ValueType>::extractLowerDTensor(const int index, Tensor *&pTensor) {
    vector<long> newDims;
    newDims = m_dims;
    newDims.erase(newDims.begin());
    if (1 == newDims.size()) {
        newDims.insert(newDims.begin(), 1);
    }
    pTensor = new Tensor<ValueType>(newDims);
    long N = pTensor->getLength();
    copyDataTo(pTensor, index * N, N);
}


//convolution or cross-correlation
template<class ValueType>
float Tensor<ValueType>::conv(const Tensor &right, int nThreads) const {
    assert(sameLength(m_dims, right.getDims()));
    const long N = getLength();
    if (nThreads > N) {
        nThreads = N;
    }
    float sum = 0.0;

    const long NRange = (N + nThreads -1)/nThreads;
    if (1 == nThreads) {
        for (long i = 0; i < N; ++i) {
            sum += e(i) * right.e(i);
        }
    }
    else {
        float *partSum = new float[nThreads];
        vector<std::thread> threadVec;
        for (int t = 0; t < nThreads; ++t) {
            threadVec.push_back(thread([this, N, partSum, t, &right, NRange]() {
                                           partSum[t] = 0;
                                           for (long i = NRange * t; i < NRange * (t + 1) && i < N; ++i) {
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
    float sum = 0.0;
    long N = getLength();
    if (nThreads > N) {
        nThreads = N;
    }
    const long NRange = (N + nThreads -1)/nThreads;
    if (1 == nThreads) {
        for (long i = 0; i < N; ++i) {
            sum += e(N - i - 1) * right.e(i);
        }
    }
    else {
        float *partSum = new float[nThreads];
        vector<std::thread> threadVec;
        for (int t = 0; t < nThreads; ++t) {
            threadVec.push_back(thread([this, N, partSum, t, &right, NRange]() {
                                           partSum[t] = 0;
                                           for (long i = NRange * t; i < NRange * (t + 1) && i < N; ++i) {
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
    long N = getLength();
    long M = N / 2;
    ValueType temp = 0;
    for (int i = 0; i < M; ++i) {
        temp = e(i);
        e(i) = e(N - 1 - i);
        e(N - 1 - i) = temp;
    }
    return *this;
}
