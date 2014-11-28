#ifndef TRAINEXAMPLES_H
#define TRAINEXAMPLES_H

//#define RANDOM_COLLECT_FIRST_STAGE

#include "common.h"

struct TrainExamples {
public:
	static int totalWeak;
	static int indexNegTrain;
	static float factorTrain;
	static int iTrain;
	static int jTrain;

	int nowStage;
	float *weight;
	vector<pair< intImage, intImage> > *samples;

	vector< pair<float, short> > *ascendingFeatures;
	TrainExamples(int __nowStage, vector< pair<intImage, intImage> > *blackList, cascadeClassifier &cascade);
	~TrainExamples();
	weakClassifier adaboost();
	vector< pair<intImage, intImage> >* calcEmpiricalError(float &falsePositive, float &detectionRate, bool recordBlackList, cascadeClassifier &cascade);
private:
	void getPositiveExamples();
	void getNegativeExamples(vector< pair<intImage, intImage> > *blackList, cascadeClassifier &cascade);
	void scanNegativeImage(int &index, float &factor, int &i, int &j, cascadeClassifier &cascade);
	void loadNegativeExamples(int &index, float &factor, int &i, int &j);
};

#endif