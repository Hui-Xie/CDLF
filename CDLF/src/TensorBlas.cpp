
#include <TensorBlas.h>
#include "mkl.h"

// y = A*x+ b
// y = A'*x + b
// if  y == nullptr, b = A*x +b;
// if  y == nullptr, b = A'*x +b;
void gemv(const bool ATranspose, const Tensor<float>* pA, const Tensor<float>* px, Tensor<float>* pb, Tensor<float>* py){
   float *pAblas, *pxblas, *pbblas;
   pAblas = (float*) mkl_calloc(pA->getLength(), sizeof(float), 64);  //initialize 0
   pxblas = (float*) mkl_calloc(px->getLength(), sizeof(float), 64);
   pbblas = (float*) mkl_calloc(pb->getLength(), sizeof(float), 64);

   //copy value
   memcpy(pAblas, pA->getData(),pA->getLength()*sizeof(float));
   memcpy(pxblas, px->getData(),px->getLength()*sizeof(float));
   memcpy(pbblas, pb->getData(),pb->getLength()*sizeof(float));

   //compute
   CBLAS_TRANSPOSE trans;
   if (ATranspose) {
       trans = CblasTrans;
   }
   else {
       trans = CblasNoTrans;
   }
   int m = pA->getDims()[0];
   int n = pA->getDims()[1];
   cblas_sgemv(CblasRowMajor, trans, m, n, 1.0, pAblas, n, pxblas, 1, 1.0, pbblas, 1);

   if (nullptr != py){
       py->copyDataFrom(pbblas, pb->getLength()*sizeof(float), 0);
   }
   else{
      pb->copyDataFrom(pbblas, pb->getLength()*sizeof(float), 0);
   }

   mkl_free(pAblas);
   mkl_free(pxblas);
   mkl_free(pbblas);
}
