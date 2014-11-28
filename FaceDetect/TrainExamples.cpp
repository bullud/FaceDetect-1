#include "TrainExamples.h"

int TrainExamples::totalWeak = 0;
int TrainExamples::indexNegTrain = 0;
float TrainExamples::factorTrain = 1.0f;
int TrainExamples::iTrain = 0;
int TrainExamples::jTrain = 0;

TrainExamples::TrainExamples(int __nowlayer, vector< pair<intImage, intImage> > *blackList, cascadeClassifier &cascade) {
	nowStage = __nowlayer;
	samples = new std::vector< pair<intImage, intImage> >;
	getPositiveExamples(); 
	getNegativeExamples(blackList, cascade);
	cout << "sample size = " << samples->size() << endl;
	assert(samples->size() == USE_SAM);
	times("");
	float *var = new float[samples->size()];
	for(int sam_idx = 0; sam_idx < (int)samples->size(); sam_idx++) {
		var[sam_idx] = getVariance((*samples)[sam_idx]);
	}
	int cc = 0;
	ascendingFeatures = new vector< pair<float, short> >[features.size()];

#pragma omp parallel for schedule (static)
	for(int feature_idx = 0; feature_idx < (int)features.size(); feature_idx++) {
		for(int sam_idx = 0; sam_idx < (int)samples->size(); sam_idx++) {
			ascendingFeatures[feature_idx].push_back(                                      //normalization
				make_pair( 1.0f * (*samples)[sam_idx].first.getFeatureSum(features[feature_idx]) / var[sam_idx], (short)sam_idx ) 
				); 
		}
		sort(ascendingFeatures[feature_idx].begin(), ascendingFeatures[feature_idx].end());
#pragma omp critical 
		{
			cc++;
		}
		if(cc % 20 == 0) {
			fprintf(stderr, "%.3f%%\r", 100.0 * cc / features.size());
			fflush(stderr);
		}
	}

	delete[] var;
	times("getOrderedExamples");

	weight = new float[USE_SAM];
	for(int i = 0; i < USE_SAM; i++) {
		weight[i] = 1.0f / USE_SAM;
	}
}

TrainExamples::~TrainExamples() {
	delete[] weight;
	delete samples;
	delete[] ascendingFeatures;
}

void TrainExamples::getPositiveExamples() {
	times("");
	int *temp = new int[TOTAL_POS];
	for(int i = 0; i < TOTAL_POS; i++) {
		temp[i] = i;
	}
	random_shuffle(temp, temp + TOTAL_POS);
	char buf[100];
	int use_pos = USE_POS;
	int cc = 0;
	for(int i = 0; i < use_pos; i++) {
		IplImage* img = cvLoadImage((pos_path + _itoa(temp[i], buf, 10) + ".png").c_str(), CV_LOAD_IMAGE_UNCHANGED);
		if(img == NULL) {
			use_pos++;
			printf("%s does not exist\n", (pos_path + _itoa(temp[i], buf, 10) + ".png").c_str());
			continue;
		}
		samples->push_back( buildIntImagePair(img) );
		cvReleaseImage(&img);
		cc++;
		fprintf(stderr, "%.3f%%\r", 100.0 * cc / USE_POS);
		fflush(stderr);
	}
	delete[] temp;
	assert(samples->size() == USE_POS);
	times("getPositiveExamples");
}

void TrainExamples::loadNegativeExamples(int &index, float &factor, int &i, int &j) {
	char path[100];
	char buf[100];
	sprintf_s(path, sizeof(path) / sizeof(char), "record_fp\\train_fp_%d.txt", nowStage);
	if( (_access( path, 0 )) != -1 ) {
		fstream input_file;
		input_file.open(path);
		int neg_index, neg_i, neg_j;
		float neg_factor;
		while(input_file >> neg_index >> neg_factor >> neg_i >> neg_j) {
			index = neg_index;
			factor = neg_factor;
			i = neg_i;
			j = neg_j;
			IplImage* img = cvLoadImage((neg_path + _itoa(neg_index, buf, 10) + ".png").c_str(), CV_LOAD_IMAGE_UNCHANGED);
			if(img == NULL) {
				printf("%s does not exist\n", (neg_path + _itoa(neg_index, buf, 10) + ".png").c_str());
				continue;
			}
			int win_size = (int) (neg_factor * FeatureSize);
			IplImage *sub_img = cvCreateImage(cvSize(win_size, win_size), IPL_DEPTH_8U, 1);
			int temp_height = neg_i + win_size;
			int temp_width = neg_j + win_size;
#pragma omp parallel for schedule (static)
			for(int x = neg_i; x < temp_height; x++) {
				for(int y = neg_j; y < temp_width; y++) {
					cvSet2D(sub_img, x - neg_i, y - neg_j, cvGet2D(img, x, y));
				}
			}

			/*
			const char *pstrWindowsTitle = "My Face Detection";
			cvNamedWindow(pstrWindowsTitle, CV_WINDOW_AUTOSIZE);
			cvShowImage(pstrWindowsTitle, sub_img);
			cvWaitKey(0);
			cvDestroyWindow(pstrWindowsTitle);
			*/

			IplImage *downscale = cvCreateImage(cvSize(20, 20), sub_img->depth, sub_img->nChannels);
			cvResize(sub_img, downscale, CV_INTER_LINEAR);
			samples->push_back(buildIntImagePair(downscale));
			//int step = max( (int) factor, 2);
			//int step = win_size / 2;
			//int step = 2;

			int step;
			if(nowStage < 9) step = max( (int) factor, 2);
			else step = 2;

			j += step;
			if(j + win_size > img->width) {
				j = 0;
				i += step;
				if(i + win_size > img->height) {
					i = 0;
					factor *= 1.05f;
					int temp_size = (int) (factor * FeatureSize);
					if(temp_size > img->height || temp_size > img->width) {
						factor = 1.0;
						index++;
					}
				}
			}
			cvReleaseImage(&sub_img);
			cvReleaseImage(&downscale);
			cvReleaseImage(&img);
			if(samples->size() == USE_SAM) {
				input_file.close();
				return;
			}
		}
		input_file.close();
	}
}

void TrainExamples::scanNegativeImage(int &index, float &factor, int &i, int &j, cascadeClassifier &cascade) {
	loadNegativeExamples(index, factor, i, j);
	__int64 discard1 = 0, discard2 = 0;
	char buf[100];
	int cnt_neg = (int)samples->size() - USE_POS;
	if(cnt_neg == USE_NEG) return;

	char path[100];
	sprintf_s(path, sizeof(path) / sizeof(char), "record_fp\\train_fp_%d.txt", nowStage);
	fstream output_file(path, ios::out | ios::app); 
	int cc = 0;
	for(; ; index++, factor = 1.0) {
		IplImage* img = cvLoadImage((neg_path + _itoa(index, buf, 10) + ".png").c_str(), CV_LOAD_IMAGE_UNCHANGED);

		if(img == NULL) { // a dead loop, until add the negative sample

			//printf("%s does not exist\n", (neg_path + _itoa(index, buf, 10) + ".png").c_str());
			fprintf(stderr, "%s does not exist %d\n", (neg_path + _itoa(index, buf, 10) + ".png").c_str(), cc++);
			fflush(stderr);
			index--;
			continue;
		}

		pair<intImage, intImage> sum = buildIntImagePair(img);
		for(; ; factor *= 1.05f, i = 0) {
			int win_size = (int) (factor * FeatureSize);
			if(win_size > img->height || win_size > img->width) {
				break;
			}
			setImageForCascadeClassifier(0, cascade, factor, win_size, sum.first, sum.second);
			//int step = max( (int) factor, 2);
			//int step = win_size / 2;
			//int step = 2;
			
			int step;
			if(nowStage < 9) step = max( (int) factor, 2);
			else step = 2;


			for(; i + win_size <= img->height; i += step, j = 0) {
				for(; j + win_size <= img->width; j += step) {
					if(judge(0, cascade, i, j, win_size, sum.first.width) == POSITIVE) {
						IplImage *sub_img = cvCreateImage(cvSize(win_size, win_size), IPL_DEPTH_8U, 1);
						int temp_height = i + win_size;
						int temp_width = j + win_size;
#pragma omp parallel for schedule (static)
						for(int x = i; x < temp_height; x++) {
							for(int y = j; y < temp_width; y++) {
								cvSet2D(sub_img, x - i, y - j, cvGet2D(img, x, y));
							}
						}

						IplImage *downscale = cvCreateImage(cvSize(20, 20), sub_img->depth, sub_img->nChannels);
						cvResize(sub_img, downscale, CV_INTER_LINEAR);
						pair<intImage, intImage> sub_sum = buildIntImagePair(downscale);
						
						
						int *tp0, *tp1, *tp2, *tp3;
						int *tpq0, *tpq1, *tpq2, *tpq3;
						int **p;
						p = new int*[TrainExamples::totalWeak * 13];
						tp0 = cascade.p0;
						tp1 = cascade.p1;
						tp2 = cascade.p2;
						tp3 = cascade.p3;
						tpq0 = cascade.pq0;
						tpq1 = cascade.pq1;
						tpq2 = cascade.pq2;
						tpq3 = cascade.pq3;
						for(int stage_idx = 0, idx = 0; stage_idx < (int)cascade.classifier.size(); stage_idx++) {
							for(int weak_idx = 0; weak_idx < (int)cascade.classifier[stage_idx].size(); weak_idx++, idx++) {
								weakClassifier &wk = cascade.classifier[stage_idx][weak_idx];
								p[idx * 12     ] = wk.p0[0];
								p[idx * 12 + 1 ] = wk.p1[0];
								p[idx * 12 + 2 ] = wk.p2[0];
								p[idx * 12 + 3 ] = wk.p3[0];

								p[idx * 12 + 4 ] = wk.p0[1];
								p[idx * 12 + 5 ] = wk.p1[1];
								p[idx * 12 + 6 ] = wk.p2[1];
								p[idx * 12 + 7 ] = wk.p3[1];

								p[idx * 12 + 8 ] = wk.p0[2];
								p[idx * 12 + 9 ] = wk.p1[2];
								p[idx * 12 + 10] = wk.p2[2];
								p[idx * 12 + 11] = wk.p3[2];
							}
						}
						
						cvReleaseImage(&sub_img);
						cvReleaseImage(&downscale);
						setImageForCascadeClassifier(0, cascade, 1, FeatureSize, sub_sum.first, sub_sum.second);
						if(judge(0, cascade, 0, 0, FeatureSize, sub_sum.first.width) == POSITIVE) {
							output_file << index << ' ' << factor << ' ' << i << ' ' << j << endl;
							samples->push_back(sub_sum);
							cnt_neg++;
							fprintf(stderr, "collect = %.3f%%  use = %.3f%%\r", 100.0 * cnt_neg / USE_NEG, 100.0 * index / TOTAL_NEG);
							fflush(stderr);
							if(cnt_neg == USE_NEG) {
								printf("discard1 = %I64d   discard2 = %I64d   toltal = %I64d   use = %.3f%%\n", 
									discard1, discard2, discard2 + discard1, 100.0 * index / TOTAL_NEG);
								cvReleaseImage(&img);
								delete[] p;
								output_file.close();
								return;

							}
						} 
						else {
							discard2++;
						}
						
						cascade.p0 = tp0;
						cascade.p1 = tp1;
						cascade.p2 = tp2;
						cascade.p3 = tp3;
						cascade.pq0 = tpq0;
						cascade.pq1 = tpq1;
						cascade.pq2 = tpq2;
						cascade.pq3 = tpq3;
						for(int stage_idx = 0, idx = 0; stage_idx < (int)cascade.classifier.size(); stage_idx++) {
							for(int weak_idx = 0; weak_idx < (int)cascade.classifier[stage_idx].size(); weak_idx++, idx++) {
								weakClassifier &wk = cascade.classifier[stage_idx][weak_idx];
			
								wk.p0[0] = p[idx * 12     ];
								wk.p1[0] = p[idx * 12 + 1 ];
								wk.p2[0] = p[idx * 12 + 2 ];
								wk.p3[0] = p[idx * 12 + 3 ];

								wk.p0[1] = p[idx * 12 + 4 ];
								wk.p1[1] = p[idx * 12 + 5 ];
								wk.p2[1] = p[idx * 12 + 6 ];
								wk.p3[1] = p[idx * 12 + 7 ];

								wk.p0[2] = p[idx * 12 + 8 ];
								wk.p1[2] = p[idx * 12 + 9 ];
								wk.p2[2] = p[idx * 12 + 10];
								wk.p3[2] = p[idx * 12 + 11];
							}
						}
						delete[] p;

					}
					else {
						discard1++;
					}
				}
			}
		}
		cvReleaseImage(&img);
	}
	errorExit("negative sample is not enough");
}

void TrainExamples::getNegativeExamples(vector< pair<intImage, intImage> > *blackList, cascadeClassifier &cascade) {

#ifdef RANDOM_COLLECT_FIRST_STAGE
	if(nowStage == 0) {
		times("");
		int ROUND;

		if(TOTAL_NEG >= 1000) {
			ROUND = 1000;
		}
		else if(TOTAL_NEG >= 100) {
			ROUND = 100;
		}
		else {
			ROUND = 10;
		}
		int eyery = USE_NEG / ROUND;
		int left = USE_NEG % ROUND;
		int tol = 0;
		for(int i = 0; i < ROUND; i++) {
			int cnt = (i == 0 ? left : 0) + eyery;
			int idx = rand() % TOTAL_NEG;
			char path[100];
			while(_access( (neg_path + _itoa(idx, path, 10) + ".png").c_str(), 0) == -1) {// not exist
				idx = rand() % TOTAL_NEG;
			}

			IplImage* img = cvLoadImage( (neg_path + _itoa(idx, path, 10) + ".png").c_str() , CV_LOAD_IMAGE_UNCHANGED);
			for(int j = 0; j < cnt; j++) {
				float max_rate = min(img->height * 1.0f / FeatureSize, img->width * 1.0f / FeatureSize);
				float factor = 1.0f * rand() / RAND_MAX * max_rate;
				while(factor < 1.0f) { // to make win_size >= FeatureSize
					factor = 1.0f * rand() / RAND_MAX * max_rate;
				}
				int win_size = (int)(factor * FeatureSize);
				int sy = rand() % img->width;
				int sx = rand() % img->height;
				while(sy + win_size - 1 >= img->width || sx + win_size - 1 >= img->height) {
					sy = rand() % img->width;
					sx = rand() % img->height;
				}
				//cout << sx << ' ' << sy << ' ' << win_size << ' ' << img->height << ' ' << img->width << endl;
				int temp_height = sx + win_size;
				int temp_width = sy + win_size;
				IplImage *sub_img = cvCreateImage(cvSize(win_size, win_size), IPL_DEPTH_8U, 1);
#pragma omp parallel for schedule (static)
				for(int x = sx; x < temp_height; x++) {
					for(int y = sy; y < temp_width; y++) {
						cvSet2D(sub_img, x - sx, y - sy, cvGet2D(img, x, y));
					}
				}	
				IplImage *downscale = cvCreateImage(cvSize(20, 20), sub_img->depth, sub_img->nChannels);
				cvResize(sub_img, downscale, CV_INTER_LINEAR);
				pair<intImage, intImage> sub_sum = buildIntImagePair(downscale);
				samples->push_back(sub_sum);
				cvReleaseImage(&sub_img);
				cvReleaseImage(&downscale);
				tol++;
				fprintf(stderr, "collect = %.3f%%  use = %.3f%%\r", 100.0 * tol / USE_NEG, 100.0 * TrainExamples::indexNegTrain / TOTAL_NEG);
				fflush(stderr);
			}
			cvReleaseImage(&img);
		}
		times("getNegativeExamples");
	}
	else 
#endif
	{
		times("");
		if(blackList != NULL) {
			for(int i = 0; i < (int)blackList->size() && samples->size() < USE_SAM; i++) {
				samples->push_back((*blackList)[i]);
			}
		}
		scanNegativeImage(TrainExamples::indexNegTrain, TrainExamples::factorTrain, TrainExamples::iTrain, TrainExamples::jTrain, cascade);	
		times("getNegativeExamples");
	}
}

weakClassifier TrainExamples::adaboost() {
	float sum_pos_weight = 0.0, sum_neg_weight = 0.0;
	for(int i = 0; i < USE_SAM; i++) {
		if(i < USE_POS) sum_pos_weight += weight[i];
		else sum_neg_weight += weight[i];
	}
	float min_err = sum_pos_weight + sum_neg_weight, threshold = 0.0;
	int index_feature = 0, index_ordered_example = 0, toggle = 0, index_example = 0;
	float max_margin = -1;

#pragma omp parallel for schedule (static)
	for(int i = 0; i < (int)features.size(); i++) {
		float below_pos_weight = 0.0, below_neg_weight = 0.0;
		for(int j = 0; j < (int)ascendingFeatures[i].size(); j++) {
			//delete the duplication
			while(j + 1 < (int)ascendingFeatures[i].size() 
				&& ascendingFeatures[i][j].first == ascendingFeatures[i][j + 1].first) {
					assert(ascendingFeatures[i][j].second < USE_SAM);
					assert(weight[ascendingFeatures[i][j].second] > 0.0);
					if(ascendingFeatures[i][j].second < USE_POS) {
						below_pos_weight += weight[ ascendingFeatures[i][j].second ];
					}
					else {
						below_neg_weight += weight[ ascendingFeatures[i][j].second ];
					}
					j++;
			}

			assert(ascendingFeatures[i][j].second < USE_SAM);
			assert(weight[ascendingFeatures[i][j].second] > 0.0);
			if(ascendingFeatures[i][j].second < USE_POS) {
				below_pos_weight += weight[ ascendingFeatures[i][j].second ];
			}
			else {
				below_neg_weight += weight[ ascendingFeatures[i][j].second ];
			}
			float temp_threshold, temp_margin;
			float err1 = below_pos_weight + sum_neg_weight - below_neg_weight;
			float err2 = below_neg_weight + sum_pos_weight - below_pos_weight;
			float err = min(err1, err2);
			if(j + 1 == ascendingFeatures[i].size()) {
				temp_threshold = ascendingFeatures[i][j].first;
				temp_margin = 0;
			}
			else {
				temp_threshold = (ascendingFeatures[i][j + 1].first + ascendingFeatures[i][j].first) / 2.0f;
				temp_margin = ascendingFeatures[i][j + 1].first - ascendingFeatures[i][j].first;
			}
			if( min_err > err || (min_err == err && temp_margin > max_margin) ) {
#pragma omp critical 
				{
					min_err = err;
					index_feature = i;
					index_ordered_example = j;
					index_example = ascendingFeatures[i][j].second;
					toggle = (err == err1 ? 1 : -1);
					threshold = temp_threshold;
					max_margin = temp_margin;
				}
			}
		}
	}
	assert(min_err >= 0.0f);
	float e = min_err / (sum_pos_weight + sum_neg_weight);
	if(e >= 0.5f) {
		errorExit("can not find a weak classifier");
	}

	if(fabs(e) < 1e-10) {
		errorExit("error rate can not be zero");
	}

	float beta = e / (1.0f - e);
	assert(beta > 0.0f);
	float alpha = 0.5f * log(1.0f / beta);

	//========================udpata======================
	assert(e / (1.0f - e) > 0.0);
	float sqrt_beta = sqrt(e / (1.0f - e));
	float sqrt_inv_beta = sqrt((1.0f - e) / e);
	float sum_weight = 0.0f;
	assert(index_feature < (int)features.size());

#pragma omp parallel for schedule (static) reduction(+: sum_weight)
	for(int i = 0; i < (int)ascendingFeatures[index_feature].size(); i++) {
		int idx = ascendingFeatures[index_feature][i].second;
		assert(idx < USE_SAM);
		if(toggle == 1) {
			if(idx < USE_POS && i <= index_ordered_example || idx >= USE_POS && i > index_ordered_example) { // miss
				weight[idx] *= sqrt_inv_beta;
			}
			else { //hit
				weight[idx] *= sqrt_beta;
			}
		}
		else {
			if(idx < USE_POS && i > index_ordered_example || idx >= USE_POS && i <= index_ordered_example) { //miss
				weight[idx] *= sqrt_inv_beta;
			}
			else { //hit
				weight[idx] *= sqrt_beta;
			}
		}

		sum_weight += weight[idx];
	}

#pragma omp parallel for schedule (static)
	for(int i = 0; i < USE_SAM; i++) {
		weight[i] /= sum_weight;
	}

	TrainExamples::totalWeak++;
	return weakClassifier(features[index_feature], alpha, threshold / (FeatureSize * FeatureSize), toggle);
}

vector< pair<intImage, intImage> >* TrainExamples::calcEmpiricalError(float &falsePositive, float &detectionRate, bool recordBlackList, cascadeClassifier &cascade) {
	vector< pair<intImage, intImage> >* ret = NULL;
	if(recordBlackList) ret = new vector< pair<intImage, intImage> >;
	detectionRate = falsePositive = 0.0;
	for(int sam_idx = 0; sam_idx < (int)samples->size(); sam_idx++) {
		setImageForCascadeClassifier(nowStage, cascade, 1.0, FeatureSize, (*samples)[sam_idx].first, (*samples)[sam_idx].second);
		if(judge(nowStage, cascade, 0, 0, FeatureSize, (*samples)[sam_idx].first.width) == POSITIVE) {
			if(sam_idx < USE_POS) {
				detectionRate++;
			}
			else {
				falsePositive++;
				if(recordBlackList) {
					ret->push_back((*samples)[sam_idx]);
				}
			}
		}
	}
	detectionRate /= USE_POS;
	falsePositive /= USE_NEG;
	return ret;
}

