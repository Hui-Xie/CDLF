//
// Created by Hui Xie on 9/18/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//


#include "GNet.h"
#include <fstream>

GNet::GNet(const string& name, const string& saveDir): FeedForwardNet(name, saveDir){
    m_pGxLayer = nullptr;
    m_pInputXLayer = nullptr;
    m_pLossLayer = nullptr;
}

GNet::~GNet(){

}

void GNet::save() {
    Net::save();
    saveGNetParameters();
}

void GNet::load() {
    Net::load();
    loadGNetParameters();
}

void GNet::saveGNetParameters() {
    const string tableHead = "Name, InputXLayerID, GxLayerID, LossLayerID, \r\n";
    string filename = m_directory + "/GNetParameters.csv";
    FILE *pFile;
    pFile = fopen(filename.c_str(), "w");
    if (nullptr == pFile) {
        printf("Error: can not open  %s  file for writing.\n", filename.c_str());
        return;
    }
    fputs(tableHead.c_str(), pFile);
    fprintf(pFile, "%s, %d, %d, %d, \r\n", m_name.c_str(), m_pInputXLayer->m_id, m_pGxLayer->m_id, m_pLossLayer->m_id);
    fclose(pFile);
}

void GNet::loadGNetParameters() {
    string filename = m_directory + "/GNetParameters.csv";
    ifstream ifs(filename.c_str());;
    char name[100];
    char lineChar[100];
    int inputXLayerID = 0;
    int gxLayerID =0;
    int lossLayerID =0;

    if (ifs.good()) {
        ifs.ignore(100, '\n'); // ignore the table head
        ifs.getline(lineChar, 100, '\n');
        for (int i = 0; i < 100; ++i) {
            if (lineChar[i] == ',') lineChar[i] = ' ';
        }
        int nFills = sscanf(lineChar, "%s  %d  %d  %d  \r\n", name, &inputXLayerID, &gxLayerID, &lossLayerID);
        if (4 != nFills) {
            cout << "Error: sscanf GnetParameter in loadGNetParameters." << endl;
        }
    }
    ifs.close();
    if (string(name) != m_name ){
        cout <<"Hint: program find inconsistent GNet name."<<endl;
    }
    m_pInputXLayer = (InputLayer*) m_layers[inputXLayerID];
    m_pGxLayer = (Layer*)m_layers[gxLayerID];
    m_pLossLayer = (CrossEntropyLoss*) m_layers[lossLayerID];
}
