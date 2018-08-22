//
// Created by hxie1 on 8/22/18.
//

#ifndef CDLF_FRAMEWORK_TIPLIO_H
#define CDLF_FRAMEWORK_TIPLIO_H

#include "tipl/tipl.hpp"
#include "Tensor.h"

template <typename VoxelType, int Dimension>
class TIPLIO {
public:
    TIPLIO();
    ~TIPLIO();

    void readNIfTIFile(const string & filename, Tensor<float>*& pTensor);
    void writeNIfTIFile(const Tensor<float>* pTensor, const vector<long>& offset, const string & filename);

};


#include "TIPLIO.hpp"

#endif //CDLF_FRAMEWORK_TIPLIO_H
