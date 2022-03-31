#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <math.h>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/types_c.h"
#include <algorithm>
#include <assert.h>
#include<ctime>
#define blockZ 32
#define blockY 160
#define blockX 160
using namespace torch::indexing;

//�����ݿ�����x,y,z
typedef struct BlockLoc {
	size_t x;
	size_t y;
	size_t z;
}BL;
//std::shared_ptr<torch::jit::script::Module> module;

//�ָ���Ҫ����Ϣ
typedef struct
{
	int width;//���������ݵĿ�
	int height;//���������ݵĸ�
	int imgNum;//���������ݵĺ��
	int xSize, ySize, zSize;//
	int mSizeX, mSizeY, mSizeZ;//�п���x,y,z��С
	bool stop{ false };//���Ʒָ����
	int progress{ 0 };//�ָ������Ϣ
	std::vector<BL> blockLocList;//��¼�������п����������б����϶��㣩
	torch::Tensor outputTensor;//�������
} OctSeg;

//��������п��������굽vector��д�ĺ����࣬src_code_2d.cpp�İ汾�̵㣬��Ҫ�ص�Ԥ����Բο��Ǹ���
void CalBlockLoc(std::vector<BL>& BlockLocList, size_t nums, size_t rows, size_t cols) {
	size_t i_step = blockZ;//z�Ჽ�������п�ĺ��
	size_t j_step = blockX;//x�Ჽ�������п�Ŀ�
	size_t k_step = blockY;//y�Ჽ�������п�ĸ�
	BlockLocList.clear();
	BlockLoc loc;
	for (size_t i = 0; i < nums; i += i_step)
	{
		if (i + i_step > nums && nums%i != 0) {
			for (size_t j = 0; j < rows; j += j_step)
			{
				if (j + j_step > rows && rows%j != 0) {
					for (size_t k = 0; k < cols; k += k_step)
					{
						if (k + k_step > cols && cols%k != 0) {
							loc.x = cols - k_step;
							loc.y = rows - j_step;
							loc.z = nums - i_step;
							BlockLocList.push_back(loc);
							break;
						}
						loc.x = k;
						loc.y = rows - j_step;
						loc.z = nums - i_step;
						BlockLocList.push_back(loc);
					}
					break;
				}
				for (size_t k = 0; k < cols; k += k_step)
				{
					if (k + k_step > cols && cols%k != 0) {
						loc.x = cols-k_step;
						loc.y = j;
						loc.z = nums - i_step;
						BlockLocList.push_back(loc);
						break;
					}
					loc.x = k;
					loc.y = j;
					loc.z = nums - i_step;
					BlockLocList.push_back(loc);
				}
			}
			break;
		}
		for (size_t j = 0; j < rows; j += j_step)
		{
			if (j + j_step > rows && rows%j != 0) {
				for (size_t k = 0; k < cols; k += k_step)
				{
					if (k + k_step > cols && cols%k != 0) {
						loc.x = cols - k_step;
						loc.y = rows - j_step;
						loc.z = i;
						BlockLocList.push_back(loc);
						break;
					}
					loc.x = k;
					loc.y = rows - j_step;
					loc.z = i;
					BlockLocList.push_back(loc);
				}
				break;
			}
			for (size_t k = 0; k < cols; k += k_step)
			{
				if (k + k_step > cols && cols%k != 0) {
					loc.x = cols - k_step;
					loc.y = j;
					loc.z = i;
					BlockLocList.push_back(loc);
					break;
				}
				loc.x = k;
				loc.y = j;
				loc.z = i;
				BlockLocList.push_back(loc);
			}
		}
	}
	//std::cout << BlockLocList.size() << std::endl;
}

//��������ֵȡ��ͼƬ�飨���û�õ���
void GetBlocks(OctSeg *os, std::vector<float> data, BlockLoc loc, float* block) {
	size_t nums = os->mSizeZ;
	size_t rows = os->mSizeY;
	size_t cols = os->mSizeX;
	for (size_t i = loc.z; i < nums + loc.z; i++)
	{
		for (size_t j = loc.y; j < rows + loc.y; j++)
		{
			for (size_t k = loc.x; k < cols + loc.x; k++)
			{
				block[(i - loc.z) * rows * cols + (j - loc.y) * cols + k - loc.x] =
					data[j * os->ySize * os->xSize + i * os->xSize + k];
			}
		}
	}
}

//���ӻ�Ԥ��ͼ
cv::Mat Visualization(cv::Mat prediction_map, std::string LUT_file) {

	cv::cvtColor(prediction_map.clone(), prediction_map, CV_GRAY2BGR);
	cv::Mat label_colours = cv::imread(LUT_file, 1);
	cv::cvtColor(label_colours, label_colours, CV_RGB2BGR);
	cv::Mat output_image;
	LUT(prediction_map, label_colours, output_image);
	return output_image;
}

//����png��ʽͼƬ
void SaveImg(std::vector<cv::Mat> images, size_t total_num, std::string outpath) {
	for (size_t i = 0; i < total_num; i++)
	{
		cv::Mat visImage = Visualization(images[i], "pascal.png");
		cv::imwrite(outpath + "\\" + std::to_string(long long((i))) + ".png", visImage);
	}
}

//��ͼƬ��ת��ΪTensor��ʽ�����û�õ���
torch::Tensor TransformData(float *block, long long num, long long rows, long long cols) {
	torch::Tensor blocks;

	blocks = torch::from_blob(block, { num ,rows,cols,1 }, torch::kFloat);//torch::kByte
	//std::cout << blocks.sizes() << std::endl;
	blocks = blocks.permute({ 3,0,1,2 });
	//std::cout << blocks.sizes() << std::endl;
	blocks = torch::unsqueeze(blocks, 0);
	blocks = blocks.to(torch::kFloat);
	return blocks;
}

//��Ԥ����ƴ�ӱ���ΪMat
void TensorToMat(std::vector<cv::Mat> &images, torch::Tensor tensor, BlockLoc loc, int step) {
	cv::Mat partMat(cv::Size(160, 160), CV_8U, tensor.data_ptr());
	cv::Rect roi_rect = cv::Rect(loc.x, loc.y, 160, 160);
	partMat.copyTo(images[loc.z + step](roi_rect));
}



// ��ʼ�������͵���r xSize = 1146, ySize = 800, zSize = 1024
// xSize, ����x��ߴ�
// ySize������y��ߴ�
// zSize������z��ߴ�
//mSizeX��ģ�����������x�ߴ�
//mSizeY��ģ�����������Y�ߴ�
//mSizeZ��ģ�����������z�ߴ�
void *OS_Init(int xSize, int ySize, int zSize)
{
	OctSeg *os = new OctSeg;
	os->stop = false;
	os->xSize = xSize;
	os->ySize = ySize;
	os->zSize = zSize;
	os->width = xSize;//���㻻ά����,�������
	os->height = zSize;//���㻻ά����,�������
	os->imgNum = ySize;//���㻻ά����,�������
	os->mSizeZ = blockZ;//�п���
	os->mSizeY = blockY;//�п�߶�
	os->mSizeX = blockX;//�п���
	os->progress = 0;//�ָ������Ϣ
	os->outputTensor = torch::zeros({ os->imgNum,os->height,os->width }, torch::kByte);//���������ʼ��
	assert(os->xSize >= os->mSizeX);
	assert(os->ySize >= os->mSizeZ);
	assert(os->zSize >= os->mSizeY);
	CalBlockLoc(os->blockLocList, os->imgNum, os->height, os->width);//��¼�п����꣬�洢��os->blockLocList
	return os;
}

// �ͷ�
// width, ͼ����
// height��ͼ��߶�
// imgNum��ͼ����Ŀ
int OS_Release(void *handle)
{
	if (handle)
		delete(handle);
	return 1;
}

//�������ֵ
float mean(unsigned char a[], int n)
{
	double sum = 0.0;
	for (int i = 0; i < n; i++)
	{
		sum += static_cast<double>(a[i]) / 255.0;
	}
	return static_cast<float>(sum / (double)n * 255.0);
}

//�������׼��
float standardDev(unsigned char a[], int n, float average)
{
	double sum = 0.0;
	for (int i = 0; i < n; i++) {
		sum += (a[i] - average)*(a[i] - average) / 255.0;
	}
	return sqrt(sum / n * 255.0);
}

//�������λ��
unsigned char percentile(unsigned char a[], int xSize, int ySize, int zSize, float percent)
{
	int len = xSize * ySize*zSize;
	size_t step = 10000;//����ϵ��
	std::vector<unsigned char> a_copy;
	a_copy.reserve(len / step);

	for (size_t i = 0; i < len; i += step) {
		a_copy.emplace_back(a[i]);
	}
	std::sort(a_copy.begin(), a_copy.end());
	int index = std::floor(static_cast<float>(a_copy.size()) * percent - 1.0);
	if (index >= a_copy.size()) index = a_copy.size() - 1;
	if (index < 0) index = 0;
	std::cout << "percentile:" << static_cast<int>(a_copy[index]) << std::endl;
	return a_copy[index];
}

//��������z_score��һ��
std::vector<float> Norm(unsigned char *volIn, int xSize, int ySize, int zSize)
{
	int len = xSize * ySize*zSize;
	std::vector<float> a_copy(len, 0);
	unsigned char pos_num = percentile(volIn, xSize, ySize, zSize, 0.9);
	for (size_t i = 0; i < len; ++i) {
		if (volIn[i] <= pos_num) {
			continue;
		}
		else {
			a_copy[i] = float(volIn[i]);
		}
	}
	float average = mean(volIn, len);
	std::cout << "average:" << average << std::endl;
	float std = standardDev(volIn, len, average);
	std::cout << "std:" << std << std::endl;
	for (size_t i = 0; i < len; i++)
	{
		a_copy[i] = (a_copy[i] - average) / std;
	}
	return a_copy;

}
//��׼�������ڲ��ԣ�
std::vector<float> Norm_1(unsigned char *volIn, int xSize, int ySize, int zSize)
{
	int len = xSize * ySize*zSize;
	float mean = 0.3383;
	float std = 0.1146;
	uchar max = 0;
	std::vector<float> a_copy(len, 0);
	for (size_t i = 0; i < len; ++i) {
		if (volIn[i] > max) {
			max = volIn[i];
		}
	}
	for (size_t i = 0; i < len; ++i) {
		a_copy[i] = (volIn[i]/float(max)-mean)/std;
	}
	return a_copy;

}

// ����OCTͼ��ָ�
int OS_OctSeg(void *handle, unsigned char *volIn, unsigned char *volOut, void(*getPro)(int)=nullptr)
{
	OctSeg *os = (OctSeg *)handle;
	using torch::jit::script::Module;
	Module module = torch::jit::load("model.pt"/*"v_norm18.pt"*/);
	std::vector<float> vol = Norm(volIn, os->xSize, os->ySize, os->zSize);
	//std::vector<float> vol = Norm_1(volIn, os->xSize, os->ySize, os->zSize);
	torch::Tensor volData = torch::tensor(vol);
	volData = volData.reshape({ os->ySize,os->xSize,os->zSize });//reshapeһά���飨������ģ�
	volData = volData.permute({ 0,2,1 });//y,z,x�����˻�ά��������ģ�
	volData = volData.to(torch::kFloat);
	torch::NoGradGuard no_grad;
	std::cout << "Segmenting " << os->blockLocList.size() << " batch(es)" << std::endl;
	clock_t segStart = clock();
	for (int i = 0; i < os->blockLocList.size(); i++)
	{
		if (os->stop) {
			return -1;
		}
		int zStart = os->blockLocList[i].z;
		int yStart = os->blockLocList[i].y;
		int xStart = os->blockLocList[i].x;
		torch::Tensor input = volData.index({ Slice(zStart,zStart+os->mSizeZ),Slice(yStart,yStart + os->mSizeY) ,Slice(xStart,xStart + os->mSizeX) });
		input = torch::unsqueeze(input, 0);
		input = torch::unsqueeze(input, 0);
		input = input.to(at::kCUDA);
		torch::Tensor output = module.forward({ input }).toTensor();
		output = torch::sigmoid(output);
		output = output.squeeze();
		output = output > 0.5;
		output = output.to(torch::kU8);
		output = output.to(torch::kCPU);
		os->outputTensor.index({ Slice(zStart,zStart + os->mSizeZ),Slice(yStart,yStart + os->mSizeY) ,Slice(xStart,xStart + os->mSizeX) }) = output;
		os->progress = (((i + 1.0) / os->blockLocList.size()) * 10000);
		if (getPro != nullptr){
			getPro((((i + 1.0) / os->blockLocList.size()) * 10000));
		}
	}
	clock_t segEnd = clock();
	std::cout << "Segment time:" << (segEnd - segStart)/1000.0<< "s" << std::endl;

	//����ָ���ͼ����Ƭͼ��
	system("md output");//����output�ļ���
	printf("Start save images...\n");
	for (size_t i = 0; i < os->ySize; i++)
	{
		cv::Mat img(cv::Size(os->xSize, os->zSize), CV_8U, os->outputTensor[i].data_ptr());
		cv::Mat visImage = Visualization(img, "pascal.png");
		cv::imwrite("output\\" + std::to_string(long long((i))) + ".png", visImage);
	}
	//����ָ���ͼ����Ƭͼ��

	unsigned char * data = (unsigned char *)os->outputTensor.data_ptr();	
	clock_t assignStart = clock();
	//һά���飬��ά
	for (size_t y = 0; y < os->ySize; y++)
		for (size_t x = 0; x < os->xSize; x++)
			for (size_t z = 0; z < os->zSize; z++){
				volOut[y*os->zSize*os->xSize + x*os->zSize + z] = data[y*os->zSize*os->xSize + z * os->xSize + x];
			}
	clock_t assignEnd = clock();
	std::cout << "Assignment time: " << assignEnd - assignStart<< "ms" << std::endl;
	std::cout << "Segmentation complete !" << std::endl;
	return 0;
}

//��ֹ�ָ�
void OS_STOP(void *handle)
{
	OctSeg *os = (OctSeg *)handle;
	os->stop = true;
}

//��ȡ�ָ����
int OS_GetProgress(void *handle)
{
	OctSeg *os = (OctSeg *)handle;
	return os->progress;
}

//���cuda����
bool EnvsCheck() {
	std::cout << "CUDA:   " << torch::cuda::is_available() << std::endl;
	std::cout << "CUDNN:  " << torch::cuda::cudnn_is_available() << std::endl;
	std::cout << "GPU(s): " << torch::cuda::device_count() << std::endl;
	if (torch::cuda::is_available() && torch::cuda::cudnn_is_available()) {
		return true;
	}
	return false;
}

static void getProgress(int value)
{
	printf("Segmenting:%.2f%%\r", (float)value / 100);
}

void OutputImg(unsigned char* volOut,int xSize,int ySize,int zSize)
{
	system("md output_out");
	printf("Start save images...\n");
	unsigned char *tmpData = (unsigned char *)malloc(xSize * zSize * sizeof(unsigned char));
	for (size_t y = 0; y < ySize; y++)
	{
		for (size_t x = 0; x < xSize; x++)
		{
			for (size_t z = 0; z < zSize; z++)
			{
				tmpData[x*zSize+z] = volOut[y*xSize*zSize+x*zSize+z]*255;
			}
		}
		cv::Mat img(cv::Size(zSize,xSize), CV_8U, tmpData);
		//cv::Mat visImage = Visualization(img, "pascal.png");
		cv::imwrite("output_out\\" + std::to_string(long long((y))) + ".png", img);
	}
	std::cout << "Segmentation complete !" << std::endl;
	free(tmpData);
}
int main()
{
	EnvsCheck();
	/*int xSize = 160, ySize = 32, zSize = 160;*//*int xSize = 1146, ySize = 64, zSize = 1024;*/int xSize = 800, ySize = 32, zSize = 715;
	std::ifstream fin;
	fin.open("test.bin", std::ios::binary);
	if (!fin) {
		std::cout << "open error!" << std::endl;
		return -1;
	}
	unsigned char *data = (unsigned char *)malloc(xSize * ySize * zSize * sizeof(unsigned char));
	unsigned char *out = (unsigned char *)malloc(xSize * ySize * zSize * sizeof(unsigned char));
	fin.read((char *)data, xSize * ySize * zSize * sizeof(unsigned char));
	//OutputImg1(data, xSize, ySize, zSize);
	void *handle = OS_Init(xSize, ySize, zSize);
	OS_OctSeg(handle, data, out, &getProgress);
	OutputImg(out,xSize,ySize,zSize);
	OS_Release(handle);
	free(data);
	free(out);
	return 1;
}
