//
// Created by Hui Xie on 8/22/18.
//

#ifndef CDLF_FRAMEWORK_TIPLIO_H
#define CDLF_FRAMEWORK_TIPLIO_H


#include "Tensor.h"
#include "tipl/tipl.hpp"

template <typename VoxelType, int Dimension>
class TIPLIO {
public:
    TIPLIO();
    ~TIPLIO();

    int readNIfTIFile(const string & filename, Tensor<float>*& pTensor);
    int write3DNIfTIFile(const Tensor<float>* pTensor, const vector<int>& offset, const string & filename);
    int write2DNIfTIFile(const Tensor<float>* pTensor, const vector<int>& offset, const string & filename);

private:
    struct tipl::io::nifti_1_header m_imageHeader1;
    struct tipl::io::nifti_2_header m_imageHeader2;

};


#include "TIPLIO.hpp"

#endif //CDLF_FRAMEWORK_TIPLIO_H
