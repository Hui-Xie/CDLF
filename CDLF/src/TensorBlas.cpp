
#include <TensorBlas.h>
#include "mkl.h"

// y = A*x+ b
// y = A'*x + b
// if  y == nullptr, b = A*x +b;
// if  y == nullptr, b = A'*x +b;
void gemv(const bool tranposeA, const Tensor<float>* pA, const Tensor<float>* px, Tensor<float>* pb, Tensor<float>* py){
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
   if (tranposeA) {
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

// y = ax+y
void axpy(const float a, const Tensor<float>* px, Tensor<float>* py){
   float *pxblas, *pyblas;
   pxblas = (float*) mkl_calloc(px->getLength(), sizeof(float), 64);
   pyblas = (float*) mkl_calloc(py->getLength(), sizeof(float), 64);

   //copy value
   memcpy(pxblas, px->getData(),px->getLength()*sizeof(float));
   memcpy(pyblas, py->getData(),py->getLength()*sizeof(float));

   //compute
   int n = px->getDims()[0];
   cblas_saxpy(n, a, pxblas, 1, pyblas, 1);

   //copy back
   py->copyDataFrom(pyblas, py->getLength()*sizeof(float), 0);

   mkl_free(pxblas);
   mkl_free(pyblas);
}

// C = a*A*B+ b*C
void gemm(const float a, const bool transposeA, const Tensor<float>* pA, const bool transposeB, const Tensor<float>* pB, const float b, Tensor<float>* pC){
   //debug
   cout<<"GEMM:  C = a*A*B+ b*C"<<endl;
   cout<<"A size: "<< vector2Str(pA->getDims())<<endl;
   cout<<"B size: "<< vector2Str(pB->getDims())<<endl;
   cout<<"C size: "<< vector2Str(pC->getDims())<<endl;
   //debug




   float *pAblas, *pBblas, *pCblas;
   pAblas = (float*) mkl_calloc(pA->getLength(), sizeof(float), 64);  //initialize 0
   pBblas = (float*) mkl_calloc(pB->getLength(), sizeof(float), 64);
   pCblas = (float*) mkl_calloc(pC->getLength(), sizeof(float), 64);

   //copy value
   memcpy(pAblas, pA->getData(),pA->getLength()*sizeof(float));
   memcpy(pBblas, pB->getData(),pB->getLength()*sizeof(float));
   memcpy(pCblas, pC->getData(),pC->getLength()*sizeof(float));

   CBLAS_TRANSPOSE transA;
   CBLAS_TRANSPOSE transB;
   int m = pC->getDims()[0];
   int n = pC->getDims()[1];
   int k = pA->getDims()[1];
   if (transA){
      k = pA->getDims()[0];
   }

   int lda, ldb;
   if (transposeA) {
      transA = CblasTrans;
      lda = max(1,m);
   }
   else {
      transA = CblasNoTrans;
      lda = max(1,k);
   }
   if (transposeB) {
      transB = CblasTrans;
      ldb =max(1,k);
   }
   else {
      transB = CblasNoTrans;
      ldb =max(1,n);
   }
   //compute
   cblas_sgemm(CblasRowMajor, transA, transB,
               m, n, k,
               a, pAblas, lda,
               pBblas, ldb,
               b, pCblas, n);

   // copy back
   pC->copyDataFrom(pCblas, pC->getLength()*sizeof(float), 0);

   mkl_free(pAblas);
   mkl_free(pBblas);
   mkl_free(pCblas);

   cout<<"End of GEMM:  C = a*A*B+ b*C"<<endl<<endl<<endl;
}


// C = a*A + b*B
void matAdd(const float a, const Tensor<float>* pA, const float b, const Tensor<float>* pB, Tensor<float>* pC){
   float *pAblas, *pBblas, *pCblas;
   pAblas = (float*) mkl_calloc(pA->getLength(), sizeof(float), 64);  //initialize 0
   pBblas = (float*) mkl_calloc(pB->getLength(), sizeof(float), 64);
   pCblas = (float*) mkl_calloc(pC->getLength(), sizeof(float), 64);
   //copy value
   memcpy(pAblas, pA->getData(),pA->getLength()*sizeof(float));
   memcpy(pBblas, pB->getData(),pB->getLength()*sizeof(float));

   int m = pA->getDims()[0];
   int n = pA->getDims()[1];

   //compute
   mkl_somatadd('R', 'N', 'N',
               m, n,
               a, pAblas, n,
               b, pBblas, n,
               pCblas, n);

   // copy back
   pC->copyDataFrom(pCblas, pC->getLength()*sizeof(float), 0);

   mkl_free(pAblas);
   mkl_free(pBblas);
   mkl_free(pCblas);

}