#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <opencv/cv.h>
#include <opencv2/highgui.hpp>
#include <caffe/caffe.hpp>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
#include <omp.h>
using namespace std;

static bool cmp_desc(const std::pair<int, float> &item1, const std::pair<int, float> &item2){
        return (item1.second > item2.second);
}


void image_to_blob_need_type(cv::Mat image, float *p_data){
	int channel = image.channels();
	int height = image.rows;
	int width = image.cols;
	float mean[3]={104.,117.,123.};
	for(int c = 0; c < channel; ++c){
		for(int h = 0; h < height; ++h){
			for( int w = 0; w < width; ++w){
				int top_index =  (c * height + h ) * width + w;
				p_data[top_index] = image.at<cv::Vec3b>(h,w)[c]-mean[c];
			}
		}
	}
}


int main(int argc,char** argv )
{
string deploy(argv[1]);
string model(argv[2]);
string image_list(argv[3]);
string result_list(argv[4]);
ifstream infile(image_list.c_str(),ios::in);
ofstream outfile(result_list.c_str());
string filename;
vector<string> pic_name_vec;
while(infile>>filename)
{
	pic_name_vec.push_back(filename);
}

caffe::Net<float>* nets[10];
for(int i=0;i<10;i++){
nets[i]=new caffe::Net<float>(deploy,caffe::TEST);
nets[i]->CopyTrainedLayersFrom(model);
}
vector<string> result_str_vec(pic_name_vec.size());
  #pragma omp parallel for num_threads (10)
for(int i=0;i<pic_name_vec.size();i++){
	int cpu_id=omp_get_thread_num();
	caffe::Net<float> *net=nets[cpu_id];
	caffe::Blob<float>* input_layer = net->input_blobs()[0];
	caffe::Blob<float>* out_layer = net->output_blobs()[0];
	cv::Mat image=cv::imread(pic_name_vec[i]);
	cv::Mat image_resize,image_crop;
	cv::Size dsize(320,320);
	cv::resize(image,image_resize,dsize,0,0,cv::INTER_CUBIC);
	cv::Rect rect_crop(10,10,299,299);
	image_crop=image_resize(rect_crop);
	image_to_blob_need_type(image_crop,input_layer->mutable_cpu_data());
	net->ForwardFrom(0);
	const float * begin = out_layer->cpu_data();
	std::vector<pair<int,float> > result_score;
	for(int j=0;j<50;j++){
		result_score.push_back(make_pair(j,*(begin+j)));
	}
	sort(result_score.begin(),result_score.begin()+50,cmp_desc);
	stringstream res_str;
	res_str<<pic_name_vec[i]<<" "<<result_score[0].first<<" "<<result_score[0].second<<endl;
	result_str_vec[i]=res_str.str();
	cout<<i<<endl;
}

for(int i=0;i<result_str_vec.size();i++)
{
outfile<<result_str_vec[i];
}

return 0;
}
