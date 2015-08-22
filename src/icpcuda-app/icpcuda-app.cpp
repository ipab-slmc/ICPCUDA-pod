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
  bool fast_mode;
  bool init_rotated;
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
    uint64_t loadDepthNew(cv::Mat1w & depth);

    bool init_;
    std::string directory_;


    cv::Mat1w prevImage;
    cv::Mat1w currImage;
    ICPOdometry* icpOdom;
    ICPSlowdometry* icpSlowdom;
    Eigen::Matrix4f currPose, prevPose;
    int64_t prevUtime, currUtime;


    void writeRawFile(cv::Mat1w & depth);
    void prefilterData(cv::Mat1w & depth);
    void writePngFile(cv::Mat1w & depth);

    int output_counter_;


};    

App::App(boost::shared_ptr<lcm::LCM> &lcm_, const CommandLineConfig& cl_cfg_) : 
       lcm_(lcm_), cl_cfg_(cl_cfg_){
  lcm_->subscribe("CAMERA",&App::imagesHandler,this);
  lcm_->subscribe("KINECT_FRAME",&App::kinectHandler,this);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::string dev(prop.name);
  std::cout << dev << std::endl;

  if (!cl_cfg_.process_incoming){
    directory_ = "/home/mfallon/logs/kinect/rgbd_dataset_freiburg1_desk/";

    std::string associationFile = directory_;
    associationFile.append("association.txt");
    asFile.open(associationFile.c_str());
    assert(!asFile.eof() && asFile.is_open());
  }

  icpOdom = new ICPOdometry(640, 480, 320, 240, 528, 528);
  icpSlowdom = new ICPSlowdometry(640, 480, 320, 240, 528, 528);

  currImage = cv::Mat1w (480, 640);
  currPose = Eigen::Matrix4f::Identity();
  if (cl_cfg_.init_rotated){
    currPose.topLeftCorner(3, 3) <<  0,  0, 1, -1,  0, 0, 0, -1, 0;
  }
  currUtime = 0;

  prevImage = cv::Mat1w(480, 640);
  prevPose = currPose;
  prevUtime = currUtime;

  init_ = false;
  output_counter_=0;
}

void quat_to_euler(Eigen::Quaternionf q, float& roll, float& pitch, float& yaw) {
  const float q0 = q.w();
  const float q1 = q.x();
  const float q2 = q.y();
  const float q3 = q.z();
  roll = atan2(2*(q0*q1+q2*q3), 1-2*(q1*q1+q2*q2));
  pitch = asin(2*(q0*q2-q3*q1));
  yaw = atan2(2*(q0*q3+q1*q2), 1-2*(q2*q2+q3*q3));
}


// Difference the transform and scale by elapsed time:
void getTransAsVelocityTrans(Eigen::Matrix4f currPose, int64_t currUtime, Eigen::Matrix4f prevPose, int64_t prevUtime, Eigen::Vector3f &linRate, Eigen::Vector3f &rotRate){
  double elapsed_time = (currUtime - prevUtime)*1E-6;
  Eigen::Isometry3f currPoseIso(currPose);
  Eigen::Isometry3f prevPoseIso(prevPose);
  Eigen::Isometry3f deltaPoseIso = prevPoseIso.inverse()*currPoseIso;
  linRate = deltaPoseIso.translation()/elapsed_time;
  Eigen::Quaternionf deltaRotQuat =  Eigen::Quaternionf(deltaPoseIso.rotation());
  Eigen::Vector3f deltaRot;
  quat_to_euler(deltaRotQuat, deltaRot(0), deltaRot(1), deltaRot(2));
  rotRate = deltaRot/elapsed_time;

  if (1==0){
    std::stringstream ss;
    std::cout << "Elapsed Time: " << elapsed_time  << " sec\n";
    std::cout << "RPY: " << deltaRot.transpose() <<" rad\n";
    std::cout << "RPY: " << deltaRot.transpose()*180/M_PI <<" deg\n";
    std::cout << "RPY: " << rotRate.transpose() <<" rad/s | velocity scaled\n";
    std::cout << "RPY: " << rotRate.transpose()*180/M_PI <<" deg/s | velocity scaled\n";
    std::cout << "XYZ " << linRate.transpose() << " m\n";
  }
}

bot_core::pose_t getPoseAsBotPose(Eigen::Matrix4f currPose, int64_t currUtime, Eigen::Matrix4f prevPose, int64_t prevUtime){
  Eigen::Vector3f trans_out = currPose.topRightCorner(3, 1);
  Eigen::Matrix3f rot_out = currPose.topLeftCorner(3, 3);
  Eigen::Quaternionf r_x(rot_out);

  bot_core::pose_t pose_msg;
  pose_msg.utime =   currUtime;
  pose_msg.pos[0] = trans_out[0];
  pose_msg.pos[1] = trans_out[1];
  pose_msg.pos[2] = trans_out[2];
  pose_msg.orientation[0] =  r_x.w();  
  pose_msg.orientation[1] =  r_x.x();  
  pose_msg.orientation[2] =  r_x.y();  
  pose_msg.orientation[3] =  r_x.z();  

  Eigen::Vector3f linRate, rotRate;
  getTransAsVelocityTrans(currPose, currUtime, prevPose, prevUtime, linRate, rotRate);
  pose_msg.vel[0] = linRate[0];
  pose_msg.vel[1] = linRate[1];
  pose_msg.vel[2] = linRate[2];
  pose_msg.rotation_rate[0] = rotRate[0];
  pose_msg.rotation_rate[1] = rotRate[1];
  pose_msg.rotation_rate[2] = rotRate[2];

  return pose_msg;
}


void tokenize(const std::string & str, std::vector<std::string> & tokens, std::string delimiters = " "){
  tokens.clear();

  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  std::string::size_type pos = str.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos){
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    lastPos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, lastPos);
  }
}

// Freiburg Associations file:
uint64_t App::loadDepth(cv::Mat1w & depth){
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

  for(unsigned int i = 0; i < 480; i++){
    for(unsigned int j = 0; j < 640; j++){
      depth.at<unsigned short>(i, j) /= 5;
    }
  }

  return time;
}

// Sequence of incrementing png files
uint64_t App::loadDepthNew(cv::Mat1w & depth){
  std::stringstream depthLoc;
  depthLoc << "./png/" << output_counter_ << ".png";
  depth = cv::imread(depthLoc.str(), CV_LOAD_IMAGE_ANYDEPTH);
  std::cout << depthLoc.str() << "\n";
  uint64_t time = 0;

  for(unsigned int i = 0; i < 480; i++){
    for(unsigned int j = 0; j < 640; j++){
      depth.at<unsigned short>(i, j) /= 5;
    }
  }

  return time;
}


void App::writePngFile(cv::Mat1w & depth){
  for(unsigned int i = 0; i < 480; i++){
    for(unsigned int j = 0; j < 640; j++){
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

void App::prefilterData(cv::Mat1w & depth){
    for(unsigned int i = 0; i < 480; i++){
      for(unsigned int j = 0; j < 640; j++){
        if (depth.at<unsigned short>(i, j) > 4000){
          //std::cout << depth.at<unsigned short>(i, j) << " " << i << " " << j << "\n";
          depth.at<unsigned short>(i, j) = 0;
        }
      }
    }
}

void App::writeRawFile(cv::Mat1w & depth){
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
  currUtime = msg->timestamp;

  // 1. Capture data (or read file)
  if (cl_cfg_.process_incoming){

    // 1.1 Decompress Data:
    const uint8_t* depth_data =NULL;
    uint8_t* uncompress_buffer = NULL;
    int buffer_size = 0;
    if(msg->depth.compression != kinect::depth_msg_t::COMPRESSION_NONE) {
      if(msg->depth.uncompressed_size > buffer_size) {
        buffer_size = msg->depth.uncompressed_size;
        uncompress_buffer = (uint8_t*) realloc(uncompress_buffer, buffer_size);
      }
      unsigned long dlen = msg->depth.uncompressed_size;
      int status = uncompress(uncompress_buffer, &dlen, 
          msg->depth.depth_data.data(), msg->depth.depth_data_nbytes);
      if(status != Z_OK) {
        return;
      }
      depth_data =(uint8_t*) uncompress_buffer;
    }else{
      buffer_size = msg->depth.depth_data_nbytes;
      depth_data = (uint8_t*) msg->depth.depth_data.data();
    }

    if (cl_cfg_.verbose)
      std::cout <<"Got kinect: " << currUtime << ", "
            <<"Compression: "<< (int) msg->depth.compression << ", Format: " << (int)msg->depth.depth_data_format << ", " 
            << msg->depth.depth_data_nbytes << "\n";


    memcpy(currImage.data,  depth_data, buffer_size);
    //memcpy(currImage.data,  msg->depth.depth_data.data() , msg->depth.depth_data_nbytes);
  }else{
    // currUtime = loadDepth(prevImage);
    loadDepthNew(currImage);
  }

  // 2. Write data back out
  // writePngFile(currImage);
  // prefilterData(currImage);
  // writeRawFile(currImage);

  // 3. Estimate motion:
  if (!init_){
    init_ = true;
  }else{
    output_counter_++;
    int threads = 128; // these params should be optimized using test program
    int blocks = 96;

    Eigen::Vector3f trans = currPose.topRightCorner(3, 1);
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = currPose.topLeftCorner(3, 3);
    if (cl_cfg_.fast_mode==1){
      icpOdom->initICPModel((unsigned short *)prevImage.data, 20.0f, currPose);
      icpOdom->initICP((unsigned short *)currImage.data, 20.0f);
      icpOdom->getIncrementalTransformation(trans, rot, threads, blocks);
    }else{
      icpSlowdom->initICPModel((unsigned short *)prevImage.data, 20.0f, currPose);
      icpSlowdom->initICP((unsigned short *)currImage.data, 20.0f);
      icpSlowdom->getIncrementalTransformation(trans, rot);
    }
    currPose.topLeftCorner(3, 3) = rot;
    currPose.topRightCorner(3, 1) = trans;
  }

  // 4. Pu
  bot_core::pose_t pose_msg = getPoseAsBotPose( currPose , currUtime, prevPose , prevUtime);
  lcm_->publish("POSE_BODY", &pose_msg );
  std::swap(prevImage, currImage); // second copied into first
  prevPose = currPose;
  prevUtime = currUtime;
}

int main(int argc, char **argv){
  CommandLineConfig cl_cfg;
  cl_cfg.verbose = false;
  cl_cfg.process_incoming = true;
  cl_cfg.fast_mode = true; // true is use fast mode, else use slow mode
  cl_cfg.init_rotated = false;

  ConciseArgs parser(argc, argv, "icpcuda-app");
  parser.add(cl_cfg.verbose, "v", "verbose", "Verbose printf");
  parser.add(cl_cfg.process_incoming, "i", "process_incoming", "process_incoming");
  parser.add(cl_cfg.fast_mode, "f", "fast_mode", "fast_mode (else slow)");
  parser.add(cl_cfg.init_rotated, "r", "init_rotated", "init_rotated with z forward, x right");
  parser.parse();
  
  boost::shared_ptr<lcm::LCM> lcm(new lcm::LCM);
  if(!lcm->good()){
    std::cerr <<"ERROR: lcm is not good()" <<std::endl;
  }
  App fo= App(lcm, cl_cfg);
  while(0 == lcm->handle());
}
