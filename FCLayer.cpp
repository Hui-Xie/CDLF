//
// Created by Sheen156 on 6/5/2018.
//

#include "FCLayer.h"

// y = W*x+ b
// where y and b is m-D vector, y is output vector;
//       x is n-D input vector
//       W is m*n dimensional matrix
FCLayer::FCLayer(long width, Layer* preLayer){
   m_type = "FullyConnected";
   m_width = width;
   m_n = preLayer->m_width;
   m_m = width;
}

FCLayer::~FCLayer(){

}
