#include "DTopencv.h"


DTopencv::DTopencv()
{
}
void DTopencv::extractTrainingData(int& numImages, CvMat *& trainingVectors, CvMat*& trainingLabels)
{
	FILE *fp = fopen("D:\\baiducloud\\tech\\OpenCV\\basicOCR\\data\\mnist\\train-images-idx3-ubyte\\train-images.idx3-ubyte", "rb");

	FILE *fp2 = fopen("D:\\baiducloud\\tech\\OpenCV\\basicOCR\\data\\mnist\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte", "rb");

	int magicNumber = readFlippedInteger(fp);
	int numImages2 = readFlippedInteger(fp);

	int numRows = readFlippedInteger(fp);

	int numCols = readFlippedInteger(fp);

	fseek(fp2, 0x08, SEEK_SET);

	//store all the images and labels
	int size = numRows*numCols;
	trainingVectors = cvCreateMat(numImages, size, CV_32FC1);
	trainingLabels = cvCreateMat(numImages, 1, CV_32FC1);
	//with memory in place, we read data from the files
	BYTE *temp = new BYTE[size];
	BYTE tempClass = 0;
	for (int i = 0; i<numImages; i++)
	{

		fread((void*)temp, size, 1, fp);

		fread((void*)(&tempClass), sizeof(BYTE), 1, fp2);

		trainingLabels->data.fl[i] = tempClass;

		for (int k = 0; k<size; k++)
			trainingVectors->data.fl[i*size + k] = temp[k];
	}

	//Data has been stored into the opencv matrix, it's ok to relase the file handler now
	fclose(fp);
	fclose(fp2);
	delete[] temp;

}

CvDTree* DTopencv::mnist_create_dtree(const CvMat* data, const CvMat* missing,
	const CvMat* responses, float p_weight){
	CvDTree* dtree;
	CvMat* var_type;
	int i, hr1 = 0, hr2 = 0, p_total = 0;
	/*float priors[] = { 1, p_weight };*/
	float *priors = 0;

	var_type = cvCreateMat(data->cols + 1, 1, CV_8U);
	cvSet(var_type, cvScalarAll(CV_VAR_CATEGORICAL)); // all the variables are categorical

	dtree = new CvDTree;

	dtree->train(data, CV_ROW_SAMPLE, responses, 0, 0, var_type, missing,
		CvDTreeParams(80, // max depth
		10, // min sample count
		0, // regression accuracy: N/A here
		true, // compute surrogate split, as we have missing data
		2, // max number of categories (use sub-optimal algorithm for larger numbers)
		10, // the number of cross-validation folds
		true, // use 1SE rule => smaller tree
		true, // throw away the pruned tree branches
		priors // the array of priors, the bigger p_weight, the more attention
		// to the poisonous mushrooms
		// (a mushroom will be judjed to be poisonous with bigger chance)
		));

	// compute hit-rate on the training database, demonstrates predict usage.
	for (i = 0; i < data->rows; i++)
	{
		CvMat sample, mask;
		cvGetRow(data, &sample, i);
		double r = dtree->predict(&sample)->value;
		if (responses->data.fl[i] != r)
		{
			hr1++;
		}
	}
	printf("\nSample Error Rate: %.1f%%", (double)hr1 * 100 / (double)data->rows);

	cvReleaseMat(&var_type);

	return dtree;

}
void DTopencv::test()
{
	printf("Use decision tree learning algorithm to recognize handwritten digit\n");
	//number of taining samples to be used
	int numImages = 6000;
	CvMat *trainingVectors = 0;
	CvMat *trainingLabels = 0;
	extractTrainingData(numImages, trainingVectors, trainingLabels);
	//Do the training
	CvDTree* dtree;
	dtree = mnist_create_dtree(trainingVectors, 0, trainingLabels,0 );
	
	cvReleaseMat(&trainingVectors);
	cvReleaseMat(&trainingLabels);

	//Recognition: Using DT

	//test number to be used
	numImages = 1000;//for tesing number
	CvMat *testVectors = 0;
	CvMat *actualLabels = 0;
	extractTestingData(numImages, testVectors,  actualLabels);
	int err = 0;
	for (int i = 0; i < testVectors->rows; i++)
	{
		CvMat sample;
		cvGetRow(testVectors, &sample, i);
		double r = dtree->predict(&sample)->value;
		if (actualLabels->data.fl[i] != r)
		{
			err++;
		}
	}
	printf("\nError Rate: %.1f%%", (double)err * 100 / (double)testVectors->rows);

	print_variable_importance(dtree);
	delete dtree;
	cvReleaseMat(&testVectors);
	cvReleaseMat(&actualLabels);
	return;



}
void DTopencv::extractTestingData(int& numImages, CvMat*&testVectors, CvMat*& actualLabels)
{
	FILE *fp = fopen("D:\\baiducloud\\tech\\OpenCV\\basicOCR\\data\\mnist\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte", "rb");
	FILE *fp2 = fopen("D:\\baiducloud\\tech\\OpenCV\\basicOCR\\data\\mnist\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte", "rb");

	int magicNumber = readFlippedInteger(fp);
	int numImages2 = readFlippedInteger(fp);
	int numRows = readFlippedInteger(fp);

	int numCols = readFlippedInteger(fp);
	int size = numRows*numCols;

	fseek(fp2, 0x08, SEEK_SET);
	testVectors = cvCreateMat(numImages, size, CV_32FC1);
	actualLabels = cvCreateMat(numImages, 1, CV_32FC1);

	//create some temporary variables
	BYTE *temp = new BYTE[size];
	BYTE tempClass = 1;
	

	//due to time consideration, test only a portion of the test 
	for (int i = 0; i<numImages; i++)
	{

		fread((void*)temp, size, 1, fp);

		fread((void*)(&tempClass), sizeof(BYTE), 1, fp2);

		actualLabels->data.fl[i] = (float)tempClass;

		for (int k = 0; k<size; k++)
		{
			testVectors->data.fl[i*size + k] = temp[k];
		}

	}
	//release the file handle as all file data extraction are done
	fclose(fp);
	fclose(fp2);
	delete[] temp;
}

int DTopencv::readFlippedInteger(FILE *fp)
{
	int ret = 0;

	BYTE *temp;

	temp = (BYTE*)(&ret);
	fread(&temp[3], sizeof(BYTE), 1, fp);
	fread(&temp[2], sizeof(BYTE), 1, fp);
	fread(&temp[1], sizeof(BYTE), 1, fp);

	fread(&temp[0], sizeof(BYTE), 1, fp);
	return ret;
}
//static const char* var_desc[] =
//{
//	"cap shape (bell=b,conical=c,convex=x,flat=f)",
//	"cap surface (fibrous=f,grooves=g,scaly=y,smooth=s)",
//	"cap color (brown=n,buff=b,cinnamon=c,gray=g,green=r,\n\tpink=p,purple=u,red=e,white=w,yellow=y)",
//	"bruises? (bruises=t,no=f)",
//	"odor (almond=a,anise=l,creosote=c,fishy=y,foul=f,\n\tmusty=m,none=n,pungent=p,spicy=s)",
//	"gill attachment (attached=a,descending=d,free=f,notched=n)",
//	"gill spacing (close=c,crowded=w,distant=d)",
//	"gill size (broad=b,narrow=n)",
//	"gill color (black=k,brown=n,buff=b,chocolate=h,gray=g,\n\tgreen=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y)",
//	"stalk shape (enlarging=e,tapering=t)",
//	"stalk root (bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r)",
//	"stalk surface above ring (ibrous=f,scaly=y,silky=k,smooth=s)",
//	"stalk surface below ring (ibrous=f,scaly=y,silky=k,smooth=s)",
//	"stalk color above ring (brown=n,buff=b,cinnamon=c,gray=g,orange=o,\n\tpink=p,red=e,white=w,yellow=y)",
//	"stalk color below ring (brown=n,buff=b,cinnamon=c,gray=g,orange=o,\n\tpink=p,red=e,white=w,yellow=y)",
//	"veil type (partial=p,universal=u)",
//	"veil color (brown=n,orange=o,white=w,yellow=y)",
//	"ring number (none=n,one=o,two=t)",
//	"ring type (cobwebby=c,evanescent=e,flaring=f,large=l,\n\tnone=n,pendant=p,sheathing=s,zone=z)",
//	"spore print color (black=k,brown=n,buff=b,chocolate=h,green=r,\n\torange=o,purple=u,white=w,yellow=y)",
//	"population (abundant=a,clustered=c,numerous=n,\n\tscattered=s,several=v,solitary=y)",
//	"habitat (grasses=g,leaves=l,meadows=m,paths=p\n\turban=u,waste=w,woods=d)",
//	0
//};
void DTopencv::print_variable_importance(CvDTree* dtree)
{
	//const CvMat* var_importance = dtree->get_var_importance();
	Mat var_importance = dtree->getVarImportance();
	int i;
	char input[1000];
	Mat importancesort;

	if (var_importance.cols == 0)
	{
		printf("Error: Variable importance can not be retrieved\n");
		return;
	}
	sortIdx(var_importance, importancesort, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);

	double var_improtatacne_total = 0;

	for (int i = 0; i < importancesort.rows; i++){
		for (int j = 0; j < 80; j ++ ){
			int index = importancesort.at<int>(i, j);
			double val = var_importance.at<double>(0, index);
			printf("Bit %d, %d: %g%%\n", index / 28 + 1, index % 28, val*100.);
		}
	}

	//for (i = 0; i < importancesort.cols*importancesort.rows; i++)
	//{
	//	int index = importancesort.step
	//	double val = var_importance->data.db[importancesort.at(];
	//	/*char buf[100];
	//	int len = (int)(strchr(var_desc[i], '(') - var_desc[i] - 1);
	//	strncpy(buf, var_desc[i], len);
	//	buf[len] = '\0';*/
	//	//printf("%s", buf);
	//	var_improtatacne_total = var_improtatacne_total + val;
	//	printf("Bit %d, %d: %g%%\n", i / 28 + 1, i % 28, val*100.);
	//}


	//for (i = 0; i < var_importance.cols*var_importance.rows; i++)
	//{
	//	double val = var_importance->data.db[i];
	//	/*char buf[100];
	//	int len = (int)(strchr(var_desc[i], '(') - var_desc[i] - 1);
	//	strncpy(buf, var_desc[i], len);
	//	buf[len] = '\0';*/
	//	//printf("%s", buf);
	//	var_improtatacne_total = var_improtatacne_total + val;
	//	printf("Bit %d, %d: %g%%\n", i/28 + 1, i%28,val*100.);
	//}
	//printf("Total variable importane%g%%\n", var_improtatacne_total*100);
}
DTopencv::~DTopencv()
{
}
