
#include "HNRadiomicsNet.h"


HNRadiomicsNet::HNRadiomicsNet(const string &name, const string &saveDir) : FeedForwardNet(name, saveDir) {
  //null
}

HNRadiomicsNet::~HNRadiomicsNet() {

}

void HNRadiomicsNet::build() {
   //null: use csv file to create network
}

void HNRadiomicsNet::train() {

}

float HNRadiomicsNet::test() {
    return 0;
}
