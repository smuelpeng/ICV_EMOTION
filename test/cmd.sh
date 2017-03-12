BOOST_INCLUDE=/home/software/local/include
BOOST_LIB=/home/software/local/lib
CUDA_INCLUDE=/home/software/cuda/8.0/include
CUDA_LIB=/home/software/cuda/8.0/lib64
g++ predict.cpp -o predict_list.exe  -I ./include -I $BOOST_INCLUDE -I $CUDA_INCLUDE  -L /home/software/local/lib/ -L ./ -lcaffe -lopencv_core -lopencv_highgui -lopencv_imgproc -fopenmp
./predict_list.exe  ../model/deploy.prototxt  ../model/ICV_EMOTION.caffemodel $1 $1_predictions
