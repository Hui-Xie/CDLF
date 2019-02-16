
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;


//TCIACode, Age, Survival(months),AliveorDead,CauseofDeath

struct Survival{
    string m_patientCode;
    int m_age;
    float  m_survivalMonth;
    bool m_isAlive;
    string m_causeOfDeath;
};


class HNClinicalDataMgr {
public:
    HNClinicalDataMgr();
    ~HNClinicalDataMgr();

    void readSurvivalData(const string & filename);

    vector<struct Survival> m_survivalVector;

};