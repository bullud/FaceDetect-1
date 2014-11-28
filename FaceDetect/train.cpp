#include "train.h"

using namespace std;

void train() {
	generateFeatures();
	cout << "total feature = " << features.size() << endl;

	cascadeClassifier cascade;
	vector< pair<intImage, intImage> > *trainingBlackList = NULL;
	vector< vector<weakClassifier> >& classifier = cascade.classifier;
	srand((unsigned int)time(NULL));
	for(int round = 0; round < MAX_STAGE; round++) {
		printf("Stage %2d:\n", round + 1);
		classifier.push_back( vector<weakClassifier>() );	

		char buf[100];
		sprintf(buf, "record_stage\\record_%d", round);
		if( (_access( buf, 0 )) != -1 ) {
			fstream input_file;
			input_file.open(buf, ios::in | ios::binary); //read

			int size;
			input_file.read( (char*)(&size), sizeof(size) );
			input_file.read( (char*)(&TrainExamples::totalWeak), sizeof(TrainExamples::totalWeak) );
			input_file.read( (char*)(&TrainExamples::indexNegTrain), sizeof(TrainExamples::indexNegTrain) );
			input_file.read( (char*)(&TrainExamples::factorTrain), sizeof(TrainExamples::factorTrain) );
			input_file.read( (char*)(&TrainExamples::iTrain), sizeof(TrainExamples::iTrain) );
			input_file.read( (char*)(&TrainExamples::jTrain), sizeof(TrainExamples::jTrain) );

			for(int i = 0; i < size; i++) {
				feature f;
				input_file.read( (char*)(&f.count_rect), sizeof(f.count_rect) );
				for(int j = 0; j < f.count_rect; j++) {
					input_file.read( (char*)(&f.rect[j].x), sizeof(f.rect[j].x) );
					input_file.read( (char*)(&f.rect[j].y), sizeof(f.rect[j].y) );
					input_file.read( (char*)(&f.rect[j].width), sizeof(f.rect[j].width) );
					input_file.read( (char*)(&f.rect[j].height), sizeof(f.rect[j].height) );
					input_file.read( (char*)(&f.weight[j]), sizeof(f.weight[j]) );
				}
				float alpha, threshold, left_val, right_val;
				int toggle;
				input_file.read( (char*)(&alpha), sizeof(alpha) );
				input_file.read( (char*)(&threshold), sizeof(threshold) );
				input_file.read( (char*)(&left_val), sizeof(left_val) );
				input_file.read( (char*)(&right_val), sizeof(right_val) );
				input_file.read( (char*)(&toggle), sizeof(toggle) );
				classifier[round].push_back( weakClassifier(f, alpha, threshold, toggle, left_val, right_val) );
			}
			puts("load");
			cout << endl << endl;
			continue;
		}



		TrainExamples trainSet(round, trainingBlackList, cascade);

		int sizeGuide = min(10 + round * 10, 200);
		float falsePositive, detectionRate;
		bool stageMissionAccomplished = false;
		while(!stageMissionAccomplished) {
			printf("after adding weak classifier %d : \n", classifier[round].size());
			classifier[round].push_back( trainSet.adaboost() );
			bool overSized = (int)classifier[round].size() >= sizeGuide ? true : false;
			bool finalTweak = overSized;
			int tweakRecord[2];
			float tweakUnit = TWEAK_UNIT;
			int tweakCounter = 0;
			float tweak = 0.0;

			//trainSet.calcEmpiricalError(falsePositive, detectionRate, false, cascade);
			//printf("before tweak:  FP = %.5f     DR = %.5f\n", falsePositive, detectionRate);

			while(abs(tweak) < 1.1) {
				for(int i = 0 ; i < (int)classifier[round].size(); i++) {
					if(classifier[round][i].toggle == 1) {
						classifier[round][i].left_val  = (tweak - 1) * classifier[round][i].alpha;
						classifier[round][i].right_val = (tweak + 1) * classifier[round][i].alpha;
					}
					else {
						classifier[round][i].left_val  = (tweak + 1) * classifier[round][i].alpha;
						classifier[round][i].right_val = (tweak - 1) * classifier[round][i].alpha;
					}
				}
				trainSet.calcEmpiricalError(falsePositive, detectionRate, false, cascade);

				if(finalTweak) {
					if(detectionRate >= DETECT_RATE && falsePositive <= FALSE_POSITIVE_RATE) { //happy break
						stageMissionAccomplished = true;
						break; 
					}
					else if(detectionRate >= DETECT_RATE) {
						break;
					}
					else {
						tweak += TWEAK_UNIT;
						continue;
					}
				}

				if(detectionRate >= DETECT_RATE && falsePositive <= FALSE_POSITIVE_RATE) { //happy break
					stageMissionAccomplished = true;
					break; 
				}
				else if(detectionRate >= DETECT_RATE && falsePositive > FALSE_POSITIVE_RATE) { //there is room to improve the false positive rate
					tweak -= tweakUnit;
					tweakCounter++;
					tweakRecord[tweakCounter & 1] = -1;
				}
				else if(detectionRate < DETECT_RATE && falsePositive <= FALSE_POSITIVE_RATE) { // there is room to improve the detection rate
					tweak += tweakUnit;
					tweakCounter++;
					tweakRecord[tweakCounter & 1] = 1;
				}
				else { //need more weak classifier
					finalTweak = true; 
					continue;
				}
				if(!finalTweak && tweakCounter > 1 && tweakRecord[0] + tweakRecord[1] == 0) {
					tweakUnit /= 2;
					tweak += tweakRecord[tweakCounter & 1] == 1 ? (-1 * tweakUnit) : tweakUnit;
					if(tweakUnit < MIN_TWEAK){
						finalTweak = true;
					}
				}
			}
			printf("after tweak:  FP = %.5f     DR = %.5f     use = %.5f%%\n\n\n", 
				falsePositive, detectionRate, 100.0 * TrainExamples::indexNegTrain / TOTAL_NEG);
		}
		if(trainingBlackList != NULL) delete trainingBlackList;
		trainingBlackList = trainSet.calcEmpiricalError(falsePositive, detectionRate, true, cascade);

		//=================== record each stage======================
		char path[100];
		sprintf(path, "record_stage\\record_%d", round);
		fstream output_file(path, ios::out | ios::trunc | ios::binary); 

		cout << "record " << round + 1 << " stage" << endl;
		int size = (int)classifier[round].size();

		output_file.write((char*)(&size), sizeof(size));
		output_file.write((char*)(&TrainExamples::totalWeak), sizeof(TrainExamples::totalWeak));
		output_file.write((char*)(&TrainExamples::indexNegTrain), sizeof(TrainExamples::indexNegTrain));
		output_file.write((char*)(&TrainExamples::factorTrain), sizeof(TrainExamples::factorTrain));
		output_file.write((char*)(&TrainExamples::iTrain), sizeof(TrainExamples::iTrain));
		output_file.write((char*)(&TrainExamples::jTrain), sizeof(TrainExamples::jTrain));
		for(int i = 0; i < (int)classifier[round].size(); i++) {
			output_file.write((char*)(&classifier[round][i].f.count_rect), sizeof(classifier[round][i].f.count_rect));
			for(int j = 0; j < classifier[round][i].f.count_rect; j++) {
				output_file.write((char*)(&classifier[round][i].f.rect[j].x), sizeof(classifier[round][i].f.rect[j].x));
				output_file.write((char*)(&classifier[round][i].f.rect[j].y), sizeof(classifier[round][i].f.rect[j].y));
				output_file.write((char*)(&classifier[round][i].f.rect[j].width), sizeof(classifier[round][i].f.rect[j].width));
				output_file.write((char*)(&classifier[round][i].f.rect[j].height), sizeof(classifier[round][i].f.rect[j].height));
				output_file.write((char*)(&classifier[round][i].f.weight[j]), sizeof(classifier[round][i].f.weight[j]));
			}
			output_file.write((char*)(&classifier[round][i].alpha), sizeof(classifier[round][i].alpha));
			output_file.write((char*)(&classifier[round][i].threshold), sizeof(classifier[round][i].threshold));
			output_file.write((char*)(&classifier[round][i].left_val), sizeof(classifier[round][i].left_val));
			output_file.write((char*)(&classifier[round][i].right_val), sizeof(classifier[round][i].right_val));
			output_file.write((char*)(&classifier[round][i].toggle), sizeof(classifier[round][i].toggle));
		}
		output_file.close();
		cout << "record " << round + 1 << " stage over" << endl;
		cout << endl << endl;
	}
	
	//==========================record a binary file==================================
	fstream output_file("record_stage\\cascade_bin", ios::out | ios::trunc | ios::binary);
	output_file.write((char*)(&MAX_STAGE), sizeof(MAX_STAGE));
	for(int round = 0; round < MAX_STAGE; round++) {
		int size = (int)classifier[round].size();
		output_file.write((char*)(&size), sizeof(size));
		for(int i = 0; i < (int)classifier[round].size(); i++) {
			output_file.write((char*)(&classifier[round][i].f.count_rect), sizeof(classifier[round][i].f.count_rect));
			for(int j = 0; j < classifier[round][i].f.count_rect; j++) {
				output_file.write((char*)(&classifier[round][i].f.rect[j].x), sizeof(classifier[round][i].f.rect[j].x));
				output_file.write((char*)(&classifier[round][i].f.rect[j].y), sizeof(classifier[round][i].f.rect[j].y));
				output_file.write((char*)(&classifier[round][i].f.rect[j].width), sizeof(classifier[round][i].f.rect[j].width));
				output_file.write((char*)(&classifier[round][i].f.rect[j].height), sizeof(classifier[round][i].f.rect[j].height));
				output_file.write((char*)(&classifier[round][i].f.weight[j]), sizeof(classifier[round][i].f.weight[j]));
			}
			output_file.write((char*)(&classifier[round][i].threshold), sizeof(classifier[round][i].threshold));
			output_file.write((char*)(&classifier[round][i].left_val), sizeof(classifier[round][i].left_val));
			output_file.write((char*)(&classifier[round][i].right_val), sizeof(classifier[round][i].right_val));
		}
	}
	output_file.close();
	//-----------------------------------------------------------------------------


	//==========================record a txt file==================================
	fstream output_file1("record_stage\\cascade_txt.txt", ios::out, ios::trunc);
	output_file1 << MAX_STAGE << endl;
	for(int round = 0; round < MAX_STAGE; round++) {
		output_file1 << classifier[round].size() << endl;
		for(int i = 0; i < (int)classifier[round].size(); i++) {
			output_file1 << classifier[round][i].f.count_rect << endl;
			for(int j = 0; j < classifier[round][i].f.count_rect; j++) {
				output_file1 << classifier[round][i].f.rect[j].x << ' '
					<< classifier[round][i].f.rect[j].y << ' '
					<< classifier[round][i].f.rect[j].width << ' '
					<< classifier[round][i].f.rect[j].height << ' '
					<< classifier[round][i].f.weight[j] << endl;
			}
			//output_file1 << setprecision(8) << classifier[round][i].alpha << endl;
			output_file1 << setprecision(8) << classifier[round][i].threshold << endl;
			output_file1 << setprecision(8) << classifier[round][i].left_val << ' ';
			output_file1 << setprecision(8) << classifier[round][i].right_val << endl;
			//output_file1 << classifier[round][i].toggle << endl;
			output_file1 << endl;
		}
	}
	output_file1.close();
	//----------------------------------------------------------------------------------

	puts("\n");
}