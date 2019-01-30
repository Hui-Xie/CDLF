#include <AssemblyLossLayer.h>


AssemblyLossLayer::AssemblyLossLayer(const int id, const string &name, Layer *prevLayer) : LossLayer(id, name, prevLayer) {
    m_type = "AssemblyLossLayer";
    m_lossList.clear();
}

AssemblyLossLayer::~AssemblyLossLayer() {
    list<LossLayer*>::iterator it = m_lossList.begin();
    while( it != m_lossList.end()){
        delete *it;
        *it = nullptr;
        ++it;
    }
    m_lossList.clear();
}

void AssemblyLossLayer::addLoss(LossLayer *lossLayer) {
   m_lossList.push_back(lossLayer);
}


float AssemblyLossLayer::lossCompute() {
    m_loss = 0.0;
    list<LossLayer*>::iterator it = m_lossList.begin();
    while( it != m_lossList.end()){
        m_loss += (*it)->lossCompute();
        ++it;
    }
    return m_loss;
}

void AssemblyLossLayer::gradientCompute() {
    list<LossLayer*>::iterator it = m_lossList.begin();
    while( it != m_lossList.end()){
        (*it)->gradientCompute();
        ++it;
    }
}


void AssemblyLossLayer::printStruct() {
    string assemblyLayer;
    list<LossLayer*>::iterator it = m_lossList.begin();
    while( it != m_lossList.end()){
        assemblyLayer += (*it)->m_type+ " ";
        ++it;
    }
    printf("id=%d, Name=%s, Type=%s, PrevLayer=%s; Assembly={%s}; \n",
           m_id, m_name.c_str(),m_type.c_str(),  m_prevLayer->m_name.c_str(), assemblyLayer.c_str());
}


