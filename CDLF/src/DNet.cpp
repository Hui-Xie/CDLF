//
// Created by Hui Xie on 9/18/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//
#include "DNet.h"
#include <fstream>


DNet::DNet(const string& saveDir):FeedForwardNet(saveDir){
    m_pGTLayer = nullptr;
    m_pGxLayer = nullptr;
    m_pInputXLayer = nullptr;
    m_pMerger = nullptr;
    m_pLossLayer = nullptr;
}

DNet::~DNet(){

}
/* [0,1]' indicate alpha = true;
 * [1,0]' indicate alpha = false;
 * */

void DNet::setAlphaGroundTruth(bool alpha){
    if (nullptr == m_pLossLayer->m_pGroundTruth){
        m_pLossLayer->m_pGroundTruth = new Tensor<float>({2,1});
    }
    Tensor<float>* pGT = m_pLossLayer->m_pGroundTruth;
    pGT->zeroInitialize();
    if (alpha) pGT->e(1) =1;
    else pGT->e(0) =1;
}

void DNet::save() {
    Net::save();
    saveDNetParameters();
}

void DNet::load() {
    Net::load();
    loadDNetParameters();
}

void DNet::saveDNetParameters() {
    const string tableHead = "Name, GTLayerID, GxLayerID, InputXLayerID, MergerID, LossLayerID, \r\n";
    string filename = m_directory + "/DNetParameters.csv";
    FILE *pFile;
    pFile = fopen(filename.c_str(), "w");
    if (nullptr == pFile) {
        printf("Error: can not open  %s  file for writing.\n", filename.c_str());
        return;
    }
    fputs(tableHead.c_str(), pFile);
    fprintf(pFile, "%s, %d, %d, %d, %d, %d, \r\n", m_name.c_str(), m_pGTLayer->m_id, m_pGxLayer->m_id, m_pInputXLayer->m_id, m_pMerger->m_id, m_pLossLayer->m_id);
    fclose(pFile);
}

void DNet::loadDNetParameters() {
    string filename = m_directory + "/DNetParameters.csv";
    ifstream ifs(filename.c_str());;
    char name[100];
    char lineChar[100];
    int gTLayerID = 0;
    int gxLayerID =0;
    int inputXLayerID = 0;
    int mergerLayerID = 0;
    int lossLayerID =0;

    if (ifs.good()) {
        ifs.ignore(100, '\n'); // ignore the table head
        ifs.getline(lineChar, 100, '\n');
        for (int i = 0; i < 100; ++i) {
            if (lineChar[i] == ',') lineChar[i] = ' ';
        }
        int nFills = sscanf(lineChar, "%s  %d  %d  %d  %d  %d  \r\n", name, &gTLayerID, &gxLayerID, &inputXLayerID, &mergerLayerID, &lossLayerID);
        if (6 != nFills) {
            cout << "Error: sscanf DnetParameters in loadDNetParameters." << endl;
        }
    }
    ifs.close();
    if (string(name) != m_name ){
        cout <<"Hint: program find inconsistent GNet name."<<endl;
    }
    m_pGTLayer = (InputLayer*) m_layers[gTLayerID];
    m_pGxLayer = (Layer*)m_layers[gxLayerID];
    m_pInputXLayer = (InputLayer*) m_layers[inputXLayerID];
    m_pMerger = ( MergerLayer*) m_layers[mergerLayerID];
    m_pLossLayer = (CrossEntropyLoss*) m_layers[lossLayerID];

}
