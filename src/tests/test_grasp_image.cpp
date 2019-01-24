#include <boost/lexical_cast.hpp>

#include <gpd/candidate/candidates_generator.h>
#include <gpd/candidate/hand.h>
#include <gpd/candidate/hand_geometry.h>
#include <gpd/descriptor/image_12_channels_strategy.h>
#include <gpd/descriptor/image_15_channels_strategy.h>
#include <gpd/descriptor/image_1_channels_strategy.h>
#include <gpd/descriptor/image_3_channels_strategy.h>
#include <gpd/descriptor/image_generator.h>
#include <gpd/net/classifier.h>
#include <gpd/util/config_file.h>
#include <gpd/util/plot.h>

namespace gpd {
namespace test {
namespace {

int DoMain(int argc, char* argv[]) {
  if (argc < 5) {
    std::cout << "ERROR: Not enough arguments given!\n";
    std::cout << "Usage: rosrun gpd test_grasp_image INPUT_FILE SAMPLE_INDEX "
                 "DRAW_GRASP_IMAGES LENET_PARAMS_DIR "
              << "[IMAGE_CHANNELS] [HAND_AXES]\n";
    return -1;
  }

  // View point from which the camera sees the point cloud.
  Eigen::Matrix3Xd view_points(3, 1);
  view_points.setZero();

  // Load point cloud from file
  std::string filename = argv[1];
  util::Cloud cloud_cam(filename, view_points);
  if (cloud_cam.getCloudOriginal()->size() == 0) {
    std::cout << "Error: Input point cloud is empty or does not exist!\n";
    return (-1);
  }

  // Read the sample index from the terminal.
  int sample_idx = boost::lexical_cast<int>(argv[2]);
  if (sample_idx >= cloud_cam.getCloudOriginal()->size()) {
    std::cout << "Error: Sample index is larger than the number of points in "
                 "the cloud!\n";
    return -1;
  }
  std::vector<int> sample_indices;
  sample_indices.push_back(sample_idx);
  cloud_cam.setSampleIndices(sample_indices);

  // Create objects to store parameters.
  candidate::CandidatesGenerator::Parameters generator_params;
  candidate::HandSearch::Parameters hand_search_params;

  // Hand geometry parameters
  candidate::HandGeometry hand_geom;
  hand_geom.finger_width_ = 0.01;
  hand_geom.outer_diameter_ = 0.12;
  hand_geom.depth_ = 0.06;
  hand_geom.height_ = 0.02;
  hand_geom.init_bite_ = 0.015;
  hand_search_params.hand_geometry_ = hand_geom;

  // Local hand search parameters
  hand_search_params.nn_radius_frames_ = 0.01;
  hand_search_params.num_orientations_ = 4;
  hand_search_params.num_samples_ = 1;
  hand_search_params.num_threads_ = 1;

  std::vector<int> hand_axes;
  if (argc >= 7) {
    for (int i = 6; i < argc; i++) {
      hand_axes.push_back(boost::lexical_cast<int>(argv[i]));
    }
  } else {
    hand_axes.push_back(2);
  }
  hand_search_params.hand_axes_ = hand_axes;
  std::cout << "hand_axes: ";
  for (int i = 0; i < hand_search_params.hand_axes_.size(); i++) {
    std::cout << hand_search_params.hand_axes_[i] << " ";
  }
  std::cout << "\n";

  // General parameters
  generator_params.num_samples_ = hand_search_params.num_samples_;
  generator_params.num_threads_ = hand_search_params.num_threads_;
  generator_params.plot_grasps_ = false;

  // Preprocessing parameters
  generator_params.remove_statistical_outliers_ = false;
  generator_params.voxelize_ = false;
  generator_params.workspace_.resize(6);
  generator_params.workspace_[0] = -1.0;
  generator_params.workspace_[1] = 1.0;
  generator_params.workspace_[2] = -1.0;
  generator_params.workspace_[3] = 1.0;
  generator_params.workspace_[4] = -1.0;
  generator_params.workspace_[5] = 1.0;

  // Image parameters
  descriptor::ImageGeometry image_params;
  image_params.depth_ = 0.06;
  image_params.height_ = 0.02;
  //  image_params.height_ = 0.1;
  image_params.outer_diameter_ = 0.12;
  image_params.size_ = 60;
  image_params.num_channels_ = 15;
  if (argc >= 6) {
    image_params.num_channels_ = boost::lexical_cast<int>(argv[5]);
  }

  // Calculate surface normals.
  cloud_cam.calculateNormals(4);
  cloud_cam.setNormals(cloud_cam.getNormals() * (-1.0));

  // Plot the normals.
  util::Plot plot(hand_search_params.hand_axes_.size(),
                  hand_search_params.num_orientations_);
  plot.plotNormals(cloud_cam.getCloudProcessed(), cloud_cam.getNormals());

  // Generate grasp candidates.
  candidate::CandidatesGenerator candidates_generator(generator_params,
                                                      hand_search_params);
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list =
      candidates_generator.generateGraspCandidateSets(cloud_cam);
  if (hand_set_list.size() == 0) {
    return -1;
  }

  plot.plotFingers3D(hand_set_list, cloud_cam.getCloudProcessed(),
                     "Grasp candidates", hand_geom, false, true);

  const candidate::Hand& hand = *hand_set_list[0]->getHands()[0];
  std::cout << "sample: " << hand.getSample().transpose() << std::endl;
  std::cout << "grasp orientation:\n" << hand.getFrame() << std::endl;
  std::cout << "grasp position: " << hand.getPosition().transpose()
            << std::endl;

  // Create the image for this grasp candidate.
  bool plot_images = true;
  plot_images = boost::lexical_cast<bool>(argv[3]);
  printf("Creating grasp image ...\n");
  printf("plot images: %d\n", plot_images);
  const int kNumThreads = 1;
  descriptor::ImageGenerator learn(image_params, kNumThreads,
                                   hand_search_params.num_orientations_,
                                   plot_images, false);
  std::vector<std::unique_ptr<cv::Mat>> images;
  std::vector<std::unique_ptr<candidate::Hand>> hands;
  learn.createImages(cloud_cam, hand_set_list, images, hands);

  plot.plotFingers3D(hands, cloud_cam.getCloudProcessed(), "Grasp candidates",
                     hand_geom);

  // Evaluate if the grasp candidates are antipodal.
  std::cout << "Antipodal: ";
  for (int i = 0; i < hands.size(); i++) {
    std::cout << hands[i]->isFullAntipodal() << " ";
  }
  std::cout << "\n";

  // Classify the grasp images.
  std::string weights_file = argv[4];
  // std::shared_ptr<net::Classifier> net =
  //     net::Classifier::create("", weights_file);
  // std::vector<float> scores2 = net->classifyImages(images);
  //
  // std::cout << "NET Scores: ";
  // for (int i = 0; i < scores2.size(); i++) {
  //   std::cout << scores2[i] << " ";
  // }
  // std::cout << "\n";
  //
  // plot.plotVolumes3D(hand_set_list, cloud_cam.getCloudProcessed(), "Volumes",
  //                    hand_geom.outer_diameter_, hand_geom.finger_width_,
  //                    hand_geom.depth_, hand_geom.height_,
  //                    image_params.outer_diameter_, image_params.depth_,
  //                    image_params.height_);

  return 0;
}

}  // namespace
}  // namespace test
}  // namespace gpd

int main(int argc, char* argv[]) { return gpd::test::DoMain(argc, argv); }
