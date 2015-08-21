#include <zlib.h>
#include <lcm/lcm-cpp.hpp>

#include <lcmtypes/bot_core.hpp>
#include <lcmtypes/kinect/frame_msg_t.hpp>
#include <boost/shared_ptr.hpp>

#include <ConciseArgs>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <icpcuda/ICPOdometry.h>
#include <icpcuda/ICPSlowdometry.h>

#include <iomanip>
#include <fstream>

using namespace std;

std::ifstream asFile;

struct CommandLineConfig
{
  bool verbose;
  bool process_incoming;
};

class App{
  public:
    App(boost::shared_ptr<lcm::LCM> &lcm_, const CommandLineConfig& cl_cfg_);
    
    ~App(){
    }

  private:
    const CommandLineConfig cl_cfg_;    
    boost::shared_ptr<lcm::LCM> lcm_;
    void imagesHandler(const lcm::ReceiveBuffer* rbuf, const std::string& channel, const  bot_core::images_t* msg);
    void kinectHandler(const lcm::ReceiveBuffer* rbuf, const std::string& channel, const  kinect::frame_msg_t* msg);

    uint64_t loadDepth(cv::Mat1w & depth);

    bool init_;
    std::string directory_;


    cv::Mat1w firstRaw;
    cv::Mat1w secondRaw;
    ICPOdometry* icpOdom;
    ICPSlowdometry* icpSlowdom;
    Eigen::Matrix4f currPose;

    uint8_t* buf_;


   void writeRawFile(cv::Mat1w & depth);
   void prefilterData(cv::Mat1w & depth);
   void writePngFile(cv::Mat1w & depth);


    int output_counter_;
};    

App::App(boost::shared_ptr<lcm::LCM> &lcm_, const CommandLineConfig& cl_cfg_) : 
       lcm_(lcm_), cl_cfg_(cl_cfg_){
  lcm_->subscribe("SCAN",&App::imagesHandler,this);
  lcm_->subscribe("KINECT_FRAME",&App::kinectHandler,this);

  directory_ = "/home/mfallon/logs/kinect/rgbd_dataset_freiburg1_desk/";
  init_ = false;

    std::string associationFile = directory_;
    associationFile.append("association.txt");

    asFile.open(associationFile.c_str());

    firstRaw = cv::Mat1w(480, 640);
    secondRaw = cv::Mat1w (480, 640);

    icpOdom = new ICPOdometry(640, 480, 320, 240, 528, 528);
    icpSlowdom = new ICPSlowdometry(640, 480, 320, 240, 528, 528);

    assert(!asFile.eof() && asFile.is_open());

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::string dev(prop.name);
  std::cout << dev << std::endl;

    currPose = Eigen::Matrix4f::Identity();


  buf_ = (uint8_t*) malloc(3*640*480);
  output_counter_=0;
}

bot_core::pose_t getPoseAsBotPose(Eigen::Isometry3f pose, int64_t utime){
  bot_core::pose_t pose_msg;
  pose_msg.utime =   utime;
  pose_msg.pos[0] = pose.translation().x();
  pose_msg.pos[1] = pose.translation().y();
  pose_msg.pos[2] = pose.translation().z();  
  Eigen::Quaternionf r_x(pose.rotation());
  pose_msg.orientation[0] =  r_x.w();  
  pose_msg.orientation[1] =  r_x.x();  
  pose_msg.orientation[2] =  r_x.y();  
  pose_msg.orientation[3] =  r_x.z();  
  return pose_msg;
}


void tokenize(const std::string & str, std::vector<std::string> & tokens, std::string delimiters = " ")
{
  tokens.clear();

  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  std::string::size_type pos = str.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos)
  {
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    lastPos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, lastPos);
  }
}

uint64_t App::loadDepth(cv::Mat1w & depth)
{
  std::string currentLine;
  std::vector<std::string> tokens;
  std::vector<std::string> timeTokens;

  getline(asFile, currentLine);
  tokenize(currentLine, tokens);

    std::cout << currentLine << "\n";

  if(tokens.size() == 0)
    return 0;

  std::string depthLoc = directory_;
  depthLoc.append(tokens[3]);
  depth = cv::imread(depthLoc, CV_LOAD_IMAGE_ANYDEPTH);

    std::cout << depthLoc << "\n";

  tokenize(tokens[0], timeTokens, ".");

  std::string timeString = timeTokens[0];
  timeString.append(timeTokens[1]);

  uint64_t time;
  std::istringstream(timeString) >> time;

  for(unsigned int i = 0; i < 480; i++)
  {
    for(unsigned int j = 0; j < 640; j++)
    {
      depth.at<unsigned short>(i, j) /= 5;
    }
  }



  ofstream myfile ("out.txt");
  if (myfile.is_open()){

    for(unsigned int i = 0; i < 480; i++){
      for(unsigned int j = 0; j < 640; j++){
        myfile << depth.at<unsigned short>(i, j);

        if (j>0)
          myfile << ", ";
      }
      myfile << "\n";
    }    
    myfile.close();

  }
  else cout << "Unable to open file";
  


  return time;
}


void App::writePngFile(cv::Mat1w & depth)
{

  for(unsigned int i = 0; i < 480; i++)
  {
    for(unsigned int j = 0; j < 640; j++)
    {
      depth.at<unsigned short>(i, j) *= 5;
    }
  }

  std::stringstream ss;
  ss << output_counter_ << ".png";

  vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);

  try {
    imwrite(ss.str(), depth, compression_params);
  }
    catch (runtime_error& ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
    return;
  }
}

void App::prefilterData(cv::Mat1w & depth)
{

    for(unsigned int i = 0; i < 480; i++){
      for(unsigned int j = 0; j < 640; j++){
        if (depth.at<unsigned short>(i, j) > 4000){
          //std::cout << depth.at<unsigned short>(i, j) << " " << i << " " << j << "\n";
          depth.at<unsigned short>(i, j) = 0;
        }
      }
    }
}

void App::writeRawFile(cv::Mat1w & depth)
{

  std::stringstream ss;
  ss << output_counter_ << ".txt";
  ofstream myfile (ss.str().c_str());
  if (myfile.is_open()){
    for(unsigned int i = 0; i < 480; i++){
      for(unsigned int j = 0; j < 640; j++){
        if (j>0)
          myfile << ", ";
        myfile << depth.at<unsigned short>(i, j);
      }
      myfile << "\n";
    }    
    myfile.close();

  }
  else cout << "Unable to open file";
}

void App::imagesHandler(const lcm::ReceiveBuffer* rbuf,
     const std::string& channel, const  bot_core::images_t* msg){
}

void App::kinectHandler(const lcm::ReceiveBuffer* rbuf,
     const std::string& channel, const  kinect::frame_msg_t* msg){
    std::cerr <<"Got kinect" <<std::endl;
  uint64_t timestamp = msg->timestamp;



  std::cout << (int) msg->depth.depth_data_format << " , ";
  std::cout << msg->depth.depth_data_nbytes << "\n";

  if (!init_){

    if (cl_cfg_.process_incoming){
      firstRaw.data = (uint8_t*) msg->depth.depth_data.data();
    }else
      loadDepth(firstRaw);

    init_ = true;
  }else{

    int threads = 128;
    int blocks = 96;

    if (cl_cfg_.process_incoming){
      memcpy(buf_,  msg->depth.depth_data.data() , msg->depth.depth_data_nbytes);
      secondRaw.data = buf_;
      //secondRaw.data = (uint8_t*) msg->depth.depth_data.data();
    }else{
      loadDepth(secondRaw);
    }


//    writePngFile(secondRaw);
  //  prefilterData(secondRaw);

    //writeRawFile(secondRaw);
    output_counter_++;




    bool mode =1;
    if (mode==1){
      icpOdom->initICPModel((unsigned short *)firstRaw.data, 20.0f, currPose);
      icpOdom->initICP((unsigned short *)secondRaw.data, 20.0f);
      Eigen::Vector3f trans = currPose.topRightCorner(3, 1);
      Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = currPose.topLeftCorner(3, 3);
      TICK("ICPFast");
      icpOdom->getIncrementalTransformation(trans, rot, threads, blocks);
      TOCK("ICPFast");
     currPose.topLeftCorner(3, 3) = rot;
      currPose.topRightCorner(3, 1) = trans;

    }else{
      icpSlowdom->initICPModel((unsigned short *)firstRaw.data, 20.0f, currPose);
      icpSlowdom->initICP((unsigned short *)secondRaw.data, 20.0f);
      Eigen::Vector3f transSlow = currPose.topRightCorner(3, 1);
      Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotSlow = currPose.topLeftCorner(3, 3);
      TICK("ICPSlow");
      icpSlowdom->getIncrementalTransformation(transSlow, rotSlow);
      TOCK("ICPSlow");
      currPose.topLeftCorner(3, 3) = rotSlow;
      currPose.topRightCorner(3, 1) = transSlow;
      Stopwatch::getInstance().sendAll();
    }
   
    std::swap(firstRaw, secondRaw);

  }


  Eigen::Vector3f trans_out = currPose.topRightCorner(3, 1);
  Eigen::Matrix3f rot_out = currPose.topLeftCorner(3, 3);
  Eigen::Quaternionf currentCameraRotation(rot_out);

  Eigen::Isometry3f tf_out;
  tf_out.setIdentity();
  tf_out.translation()  << trans_out;
  tf_out.rotate(currentCameraRotation);

  std::cout << timestamp << "\n";
  bot_core::pose_t pose_msg = getPoseAsBotPose( tf_out , timestamp);
  lcm_->publish("POSE_BODY", &pose_msg );
}

int main(int argc, char **argv){
  CommandLineConfig cl_cfg;
  cl_cfg.verbose = false;
  cl_cfg.process_incoming = true;

  ConciseArgs parser(argc, argv, "icpcuda-app");
  parser.add(cl_cfg.verbose, "v", "verbose", "Verbose printf");
  parser.add(cl_cfg.process_incoming, "i", "process_incoming", "process_incoming");
  parser.parse();
  
  boost::shared_ptr<lcm::LCM> lcm(new lcm::LCM);
  if(!lcm->good()){
    std::cerr <<"ERROR: lcm is not good()" <<std::endl;
  }
  App fo= App(lcm, cl_cfg);
  while(0 == lcm->handle());
}
