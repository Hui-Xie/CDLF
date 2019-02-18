
#include <stdio.h>
#include <string>
#include <vector>
#include <Tensor.h>

using namespace std;


//TCIACode, Age, Survival(months),AliveorDead,CauseofDeath

class Survival{
public:
    Survival();
    ~Survival();
    string m_patientCode;
    int m_age;
    float  m_survivalMonth;
    bool m_isAlive;
    string m_causeOfDeath;

    void print();
};


class HNClinicalDataMgr {
public:
    HNClinicalDataMgr(const string& clinicalFile);
    ~HNClinicalDataMgr();

    void readSurvivalData(const string & filename);

    // get a {2*years} tensor of survival rate (dead(0), alive(1)) probability
    Tensor<float>  getSurvivalTensor(const Survival&  survival, const int years);
    string getPatientCode(const string & imageFilename);
    Survival getSurvivalData(const string& patientCode);

    vector<Survival> m_survivalVector;

};