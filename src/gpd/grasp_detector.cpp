#include <gpd/grasp_detector.h>

namespace gpd {

GraspDetector::GraspDetector(const std::string& config_filename) {
  Eigen::initParallel();

  // Read parameters from configuration file.
  util::ConfigFile config_file(config_filename);
  config_file.ExtractKeys();

  // Read hand geometry parameters.
  std::string hand_geometry_filename =
      config_file.getValueOfKeyAsString("hand_geometry_filename", "");
  if (hand_geometry_filename == "0") {
    hand_geometry_filename = config_filename;
  }
  candidate::HandGeometry hand_geom(hand_geometry_filename);
  std::cout << hand_geom;

  // Read plotting parameters.
  plot_normals_ = config_file.getValueOfKey<bool>("plot_normals", false);
  plot_samples_ = config_file.getValueOfKey<bool>("plot_samples", true);
  plot_candidates_ = config_file.getValueOfKey<bool>("plot_candidates", false);
  plot_filtered_candidates_ =
      config_file.getValueOfKey<bool>("plot_filtered_candidates", false);
  plot_volumes_ = config_file.getValueOfKey<bool>("plot_volumes", false);
  plot_valid_grasps_ =
      config_file.getValueOfKey<bool>("plot_valid_grasps", false);
  plot_clustered_grasps_ =
      config_file.getValueOfKey<bool>("plot_clustered_grasps", false);
  plot_selected_grasps_ =
      config_file.getValueOfKey<bool>("plot_selected_grasps", false);
  std::cout << "============ PLOTTING ========================\n";
  std::cout << "plot_normals: " << plot_normals_ << "\n";
  std::cout << "plot_samples: " << plot_samples_ << "\n";
  std::cout << "plot_candidates: " << plot_candidates_ << "\n";
  std::cout << "plot_filtered_candidates: " << plot_filtered_candidates_
            << "\n";
  std::cout << "plot_volumes_: " << plot_volumes_ << "\n";
  std::cout << "plot_valid_grasps: " << plot_valid_grasps_ << "\n";
  std::cout << "plot_clustered_grasps: " << plot_clustered_grasps_ << "\n";
  std::cout << "plot_selected_grasps: " << plot_selected_grasps_ << "\n";
  std::cout << "==============================================\n";

  // Create object to generate grasp candidates.
  candidate::CandidatesGenerator::Parameters generator_params;
  generator_params.num_samples_ =
      config_file.getValueOfKey<int>("num_samples", 1000);
  generator_params.num_threads_ =
      config_file.getValueOfKey<int>("num_threads", 1);
  generator_params.remove_statistical_outliers_ =
      config_file.getValueOfKey<bool>("remove_outliers", false);
  generator_params.voxelize_ =
      config_file.getValueOfKey<bool>("voxelize", true);
  generator_params.workspace_ =
      config_file.getValueOfKeyAsStdVectorDouble("workspace", "-1 1 -1 1 -1 1");
  candidate::HandSearch::Parameters hand_search_params;
  hand_search_params.hand_geometry_ = hand_geom;
  hand_search_params.nn_radius_frames_ =
      config_file.getValueOfKey<double>("nn_radius", 0.01);
  hand_search_params.num_samples_ =
      config_file.getValueOfKey<int>("num_samples", 1000);
  hand_search_params.num_threads_ =
      config_file.getValueOfKey<int>("num_threads", 1);
  hand_search_params.num_orientations_ =
      config_file.getValueOfKey<int>("num_orientations", 8);
  hand_search_params.hand_axes_ =
      config_file.getValueOfKeyAsStdVectorInt("hand_axes", "2");
  candidates_generator_ = std::make_unique<candidate::CandidatesGenerator>(
      generator_params, hand_search_params);

  std::cout << "============ CLOUD PREPROCESSING =============\n";
  std::cout << "voxelize: " << generator_params.voxelize_ << "\n";
  std::cout << "remove_outliers: "
            << generator_params.remove_statistical_outliers_ << "\n";
  printStdVector(generator_params.workspace_, "workspace");

  std::cout << "============ CANDIDATE GENERATION ============\n";
  std::cout << "num_samples: " << hand_search_params.num_samples_ << "\n";
  std::cout << "num_threads: " << hand_search_params.num_threads_ << "\n";
  std::cout << "nn_radius: " << hand_search_params.nn_radius_frames_ << "\n";
  printStdVector(hand_search_params.hand_axes_, "hand axes");
  std::cout << "num_orientations: " << hand_search_params.num_orientations_
            << "\n";
  std::cout << "==============================================\n";

  // TODO: Set the camera position.
  //  Eigen::Matrix3Xd view_points(3,1);
  //  view_points << camera_position[0], camera_position[1], camera_position[2];

  // Read grasp image parameters.
  std::string image_geometry_filename =
      config_file.getValueOfKeyAsString("image_geometry_filename", "");
  if (image_geometry_filename == "0") {
    image_geometry_filename = config_filename;
  }
  descriptor::ImageGeometry image_geom(image_geometry_filename);
  std::cout << image_geom;

  // Read classification parameters and create classifier.
  std::string model_file = config_file.getValueOfKeyAsString("model_file", "");
  std::string weights_file =
      config_file.getValueOfKeyAsString("weights_file", "");
  if (!model_file.empty() || !weights_file.empty()) {
    int device = config_file.getValueOfKey<int>("device", 0);
    int batch_size = config_file.getValueOfKey<int>("batch_size", 1);
    classifier_ = net::Classifier::create(
        model_file, weights_file, static_cast<net::Classifier::Device>(device),
        batch_size);
    min_score_ = config_file.getValueOfKey<int>("min_score", 0);
    std::cout << "============ CLASSIFIER ======================\n";
    std::cout << "model_file: " << model_file << "\n";
    std::cout << "weights_file: " << weights_file << "\n";
    std::cout << "batch_size: " << batch_size << "\n";
    std::cout << "==============================================\n";
  }

  // Read additional grasp image creation parameters.
  bool remove_plane = config_file.getValueOfKey<bool>(
      "remove_plane_before_image_calculation", false);

  // Create object to create grasp images from grasp candidates (used for
  // classification).
  image_generator_ = std::make_unique<descriptor::ImageGenerator>(
      image_geom, hand_search_params.num_threads_,
      hand_search_params.num_orientations_, false, remove_plane);

  // Read grasp filtering parameters.
  workspace_grasps_ = config_file.getValueOfKeyAsStdVectorDouble(
      "workspace_grasps", "-1 1 -1 1 -1 1");
  min_aperture_ = config_file.getValueOfKey<double>("min_aperture", 0.0);
  max_aperture_ = config_file.getValueOfKey<double>("max_aperture", 0.85);
  std::cout << "============ CANDIDATE FILTERING =============\n";
  printStdVector(workspace_grasps_, "candidate_workspace");
  std::cout << "min_aperture: " << min_aperture_ << "\n";
  std::cout << "max_aperture: " << max_aperture_ << "\n";
  std::cout << "==============================================\n";

  // Read grasp filtering parameters for side grasps that are too close to the
  // table plane.
  filter_table_side_grasps_ =
      config_file.getValueOfKey<bool>("filter_table_side_grasps", false);
  vert_axis_ =
      config_file.getValueOfKeyAsStdVectorDouble("vertical_axis", "0 0 1");
  angle_thresh_ = config_file.getValueOfKey<double>("angle_thresh", 0.1);
  table_height_ = config_file.getValueOfKey<double>("table_height", 0.5);
  table_thresh_ = config_file.getValueOfKey<double>("table_thresh_", 0.05);

  // Read clustering parameters.
  int min_inliers = config_file.getValueOfKey<int>("min_inliers", 1);
  clustering_ = std::make_unique<Clustering>(min_inliers);
  cluster_grasps_ = min_inliers > 0 ? true : false;
  std::cout << "============ CLUSTERING ======================\n";
  std::cout << "min_inliers: " << min_inliers << "\n";
  std::cout << "==============================================\n\n";

  // Read grasp selection parameters.
  num_selected_ = config_file.getValueOfKey<int>("num_selected", 100);

  // Create plotter.
  plotter_ = std::make_unique<util::Plot>(hand_search_params.hand_axes_.size(),
                                          hand_search_params.num_orientations_);
}

std::vector<std::unique_ptr<candidate::Hand>> GraspDetector::detectGrasps(
    const util::Cloud& cloud) {
  double t0_total = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::Hand>> hands_out;

  const candidate::HandGeometry& hand_geom =
      candidates_generator_->getHandSearchParams().hand_geometry_;

  // Check if the point cloud is empty.
  if (cloud.getCloudOriginal()->size() == 0) {
    printf("ERROR: Point cloud is empty!");
    hands_out.resize(0);
    return hands_out;
  }

  // Plot samples/indices.
  if (plot_samples_) {
    if (cloud.getSamples().cols() > 0) {
      plotter_->plotSamples(cloud.getSamples(), cloud.getCloudProcessed());
    } else if (cloud.getSampleIndices().size() > 0) {
      plotter_->plotSamples(cloud.getSampleIndices(),
                            cloud.getCloudProcessed());
    }
  }

  if (plot_normals_) {
    std::cout << "Plotting normals for different camera sources\n";
    plotter_->plotNormals(cloud);
  }

  // 1. Generate grasp candidates.
  double t0_candidates = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list =
      candidates_generator_->generateGraspCandidateSets(cloud);
  printf("Generated %zu hand sets.\n", hand_set_list.size());
  if (hand_set_list.size() == 0) {
    return hands_out;
  }
  double t_candidates = omp_get_wtime() - t0_candidates;
  if (plot_candidates_) {
    plotter_->plotFingers3D(hand_set_list, cloud.getCloudOriginal(),
                            "Grasp candidates", hand_geom);
  }

  // 2. Filter the candidates.
  double t0_filter = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_filtered =
      filterGraspsWorkspace(hand_set_list, workspace_grasps_);
  if (plot_filtered_candidates_) {
    plotter_->plotFingers3D(hands_out, cloud.getCloudOriginal(),
                            "Filtered Grasps (Aperture, Workspace)", hand_geom);
  }
  if (filter_table_side_grasps_) {
    hand_set_list_filtered =
        filterSideGraspsCloseToTable(hand_set_list_filtered);
    if (plot_filtered_candidates_) {
      plotter_->plotFingers3D(hands_out, cloud.getCloudOriginal(),
                              "Filtered Grasps (Aperture, Workspace)",
                              hand_geom);
    }
  }
  double t_filter = omp_get_wtime() - t0_filter;

  // 3. Create grasp descriptors (images).
  double t0_images = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::Hand>> hands;
  std::vector<std::unique_ptr<cv::Mat>> images;
  image_generator_->createImages(cloud, hand_set_list_filtered, images, hands);
  double t_images = omp_get_wtime() - t0_images;

  // 4. Classify the grasp candidates.
  double t0_classify = omp_get_wtime();
  std::vector<float> scores = classifier_->classifyImages(images);
  for (int i = 0; i < hands.size(); i++) {
    hands[i]->setScore(scores[i]);
  }
  double t_classify = omp_get_wtime() - t0_classify;

  // 5. Select the <num_selected> highest scoring grasps.
  hands = selectGrasps(hands);
  if (plot_valid_grasps_) {
    plotter_->plotFingers3D(hands, cloud.getCloudOriginal(), "Valid Grasps",
                            hand_geom);
  }

  // 6. Cluster the grasps.
  double t0_cluster = omp_get_wtime();
  std::vector<std::unique_ptr<candidate::Hand>> clusters;
  if (cluster_grasps_) {
    clusters = clustering_->findClusters(hands);
    printf("Found %d clusters.\n", (int)clusters.size());
    if (clusters.size() <= 3) {
      printf(
          "Not enough clusters found! Adding all grasps from previous step.");
      for (int i = 0; i < hands.size(); i++) {
        clusters.push_back(std::move(hands[i]));
      }
    }
    if (plot_clustered_grasps_) {
      plotter_->plotFingers3D(clusters, cloud.getCloudOriginal(),
                              "Clustered Grasps", hand_geom);
    }
  } else {
    clusters = std::move(hands);
  }
  double t_cluster = omp_get_wtime() - t0_cluster;

  // 7. Sort grasps by their score.
  std::sort(clusters.begin(), clusters.end(), isScoreGreater);
  printf("======== Selected grasps ========\n");
  for (int i = 0; i < clusters.size(); i++) {
    std::cout << "Grasp " << i << ": " << clusters[i]->getScore() << "\n";
  }
  printf("Selected the %d best grasps.\n", (int)clusters.size());
  double t_total = omp_get_wtime() - t0_total;

  printf("======== RUNTIMES ========\n");
  printf(" 1. Candidate generation: %3.4fs\n", t_candidates);
  printf(" 2. Descriptor extraction: %3.4fs\n", t_images);
  printf(" 3. Classification: %3.4fs\n", t_classify);
  // printf(" Filtering: %3.4fs\n", t_filter);
  // printf(" Clustering: %3.4fs\n", t_cluster);
  printf("==========\n");
  printf(" TOTAL: %3.4fs\n", t_total);

  if (plot_selected_grasps_) {
    plotter_->plotFingers3D(clusters, cloud.getCloudOriginal(),
                            "Selected Grasps", hand_geom, false);
  }

  return clusters;
}

void GraspDetector::preprocessPointCloud(util::Cloud& cloud) {
  candidates_generator_->preprocessPointCloud(cloud);
}

std::vector<std::unique_ptr<candidate::HandSet>>
GraspDetector::filterGraspsWorkspace(
    std::vector<std::unique_ptr<candidate::HandSet>>& hand_set_list,
    const std::vector<double>& workspace) const {
  int remaining = 0;
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_out;
  printf("Filtering grasps outside of workspace ...\n");

  const candidate::HandGeometry& hand_geometry =
      candidates_generator_->getHandSearchParams().hand_geometry_;

  for (int i = 0; i < hand_set_list.size(); i++) {
    const std::vector<std::unique_ptr<candidate::Hand>>& hands =
        hand_set_list[i]->getHands();
    Eigen::Array<bool, 1, Eigen::Dynamic> is_valid =
        hand_set_list[i]->getIsValid();

    for (int j = 0; j < hands.size(); j++) {
      if (!is_valid(j)) {
        continue;
      }
      double half_width = 0.5 * hand_geometry.outer_diameter_;
      Eigen::Vector3d left_bottom =
          hands[j]->getPosition() + half_width * hands[j]->getBinormal();
      Eigen::Vector3d right_bottom =
          hands[j]->getPosition() - half_width * hands[j]->getBinormal();
      Eigen::Vector3d left_top =
          left_bottom + hand_geometry.depth_ * hands[j]->getApproach();
      Eigen::Vector3d right_top =
          left_bottom + hand_geometry.depth_ * hands[j]->getApproach();
      Eigen::Vector3d approach =
          hands[j]->getPosition() - 0.05 * hands[j]->getApproach();
      Eigen::VectorXd x(5), y(5), z(5);
      x << left_bottom(0), right_bottom(0), left_top(0), right_top(0),
          approach(0);
      y << left_bottom(1), right_bottom(1), left_top(1), right_top(1),
          approach(1);
      z << left_bottom(2), right_bottom(2), left_top(2), right_top(2),
          approach(2);

      // Ensure the object fits into the hand and avoid grasps outside the
      // workspace.
      if (hands[j]->getGraspWidth() >= min_aperture_ &&
          hands[j]->getGraspWidth() <= max_aperture_ &&
          x.minCoeff() >= workspace[0] && x.maxCoeff() <= workspace[1] &&
          y.minCoeff() >= workspace[2] && y.maxCoeff() <= workspace[3] &&
          z.minCoeff() >= workspace[4] && z.maxCoeff() <= workspace[5]) {
        is_valid(j) = true;
        remaining++;
      } else {
        is_valid(j) = false;
      }
    }

    if (is_valid.any()) {
      hand_set_list_out.push_back(std::move(hand_set_list[i]));
      hand_set_list_out[hand_set_list_out.size() - 1]->setIsValid(is_valid);
    }
  }

  printf("Number of grasps within workspace and gripper width: %d\n",
         remaining);

  return hand_set_list_out;
}

std::vector<std::unique_ptr<candidate::HandSet>>
GraspDetector::generateGraspCandidates(const util::Cloud& cloud) {
  return candidates_generator_->generateGraspCandidateSets(cloud);
}

std::vector<std::unique_ptr<candidate::Hand>> GraspDetector::selectGrasps(
    std::vector<std::unique_ptr<candidate::Hand>>& hands) const {
  printf("Selecting the %d highest scoring grasps ...\n", num_selected_);

  int middle = std::min((int)hands.size(), num_selected_);
  std::partial_sort(hands.begin(), hands.begin() + middle, hands.end(),
                    isScoreGreater);
  std::vector<std::unique_ptr<candidate::Hand>> hands_out;

  for (int i = 0; i < middle; i++) {
    hands_out.push_back(std::move(hands[i]));
    printf(" grasp #%d, score: %3.4f\n", i, hands_out[i]->getScore());
  }

  return hands_out;
}

std::vector<std::unique_ptr<candidate::HandSet>>
GraspDetector::filterSideGraspsCloseToTable(
    std::vector<std::unique_ptr<candidate::HandSet>>& hand_set_list) {
  const double APPROACH_LENGTH = 0.05;

  int remaining = 0;
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_out;
  Eigen::Vector3d vert_axis_vec;
  vert_axis_vec << vert_axis_[0], vert_axis_[1], vert_axis_[2];

  for (int i = 0; i < hand_set_list.size(); i++) {
    const std::vector<std::unique_ptr<candidate::Hand>>& hands =
        hand_set_list[i]->getHands();
    Eigen::Array<bool, 1, Eigen::Dynamic> is_valid =
        hand_set_list[i]->getIsValid();

    for (int j = 0; j < hands.size(); j++) {
      if (is_valid(j)) {
        double angle =
            fabs(vert_axis_vec.transpose() * hands[i]->getApproach());
        double dist = fabs((hands[i]->getPosition() -
                            APPROACH_LENGTH * hands[i]->getApproach())(2)) -
                      table_height_;

        // This is a side grasps that is too close to the table.
        if (angle > angle_thresh_ && dist < table_thresh_) {
          is_valid(j) = false;
        } else {
          is_valid(j) = true;
          remaining++;
        }
      }
    }

    if (is_valid.any()) {
      hand_set_list_out.push_back(std::move(hand_set_list[i]));
      hand_set_list_out[hand_set_list_out.size() - 1]->setIsValid(is_valid);
    }
  }

  printf("Number of grasps that are not too close to the table: %d\n",
         remaining);

  return hand_set_list_out;
}

bool GraspDetector::createGraspImages(
    util::Cloud& cloud,
    std::vector<std::unique_ptr<candidate::Hand>>& hands_out,
    std::vector<std::unique_ptr<cv::Mat>>& images_out) {
  // Check if the point cloud is empty.
  if (cloud.getCloudOriginal()->size() == 0) {
    printf("ERROR: Point cloud is empty!");
    hands_out.resize(0);
    images_out.resize(0);
    return false;
  }

  // Plot samples/indices.
  if (plot_samples_) {
    if (cloud.getSamples().cols() > 0) {
      plotter_->plotSamples(cloud.getSamples(), cloud.getCloudProcessed());
    } else if (cloud.getSampleIndices().size() > 0) {
      plotter_->plotSamples(cloud.getSampleIndices(),
                            cloud.getCloudProcessed());
    }
  }

  if (plot_normals_) {
    std::cout << "Plotting normals for different camera sources\n";
    plotter_->plotNormals(cloud);
  }

  // 1. Generate grasp candidates.
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list =
      candidates_generator_->generateGraspCandidateSets(cloud);
  printf("Generated %zu hand sets.\n", hand_set_list.size());
  if (hand_set_list.size() == 0) {
    hands_out.resize(0);
    images_out.resize(0);
    return false;
  }

  const candidate::HandGeometry& hand_geom =
      candidates_generator_->getHandSearchParams().hand_geometry_;

  // 2. Filter the candidates.
  std::vector<std::unique_ptr<candidate::HandSet>> hand_set_list_filtered =
      filterGraspsWorkspace(hand_set_list, workspace_grasps_);
  if (plot_filtered_candidates_) {
    plotter_->plotFingers3D(hands_out, cloud.getCloudOriginal(),
                            "Filtered Grasps (Aperture, Workspace)", hand_geom);
  }
  if (filter_table_side_grasps_) {
    hand_set_list_filtered =
        filterSideGraspsCloseToTable(hand_set_list_filtered);
    if (plot_filtered_candidates_) {
      plotter_->plotFingers3D(hands_out, cloud.getCloudOriginal(),
                              "Filtered Grasps (Aperture, Workspace)",
                              hand_geom);
    }
  }

  // 3. Create grasp descriptors (images).
  std::vector<std::unique_ptr<candidate::Hand>> hands;
  std::vector<std::unique_ptr<cv::Mat>> images;
  image_generator_->createImages(cloud, hand_set_list_filtered, images_out,
                                 hands_out);

  return true;
}

std::vector<int> GraspDetector::evalGroundTruth(
    const util::Cloud& cloud_gt,
    std::vector<std::unique_ptr<candidate::Hand>>& hands) {
  return candidates_generator_->reevaluateHypotheses(cloud_gt, hands);
}

std::vector<std::unique_ptr<candidate::Hand>>
GraspDetector::pruneGraspCandidates(
    const util::Cloud& cloud,
    const std::vector<std::unique_ptr<candidate::HandSet>>& hand_set_list,
    double min_score) {
  // 1. Create grasp descriptors (images).
  std::vector<std::unique_ptr<candidate::Hand>> hands;
  std::vector<std::unique_ptr<cv::Mat>> images;
  image_generator_->createImages(cloud, hand_set_list, images, hands);

  // 2. Classify the grasp candidates.
  std::vector<float> scores = classifier_->classifyImages(images);
  std::vector<std::unique_ptr<candidate::Hand>> hands_out;

  // 3. Only keep grasps with a score larger than <min_score>.
  for (int i = 0; i < hands.size(); i++) {
    if (scores[i] > min_score) {
      hands[i]->setScore(scores[i]);
      hands_out.push_back(std::move(hands[i]));
    }
  }

  return hands_out;
}

void GraspDetector::printStdVector(const std::vector<int>& v,
                                   const std::string& name) const {
  printf("%s: ", name.c_str());
  for (int i = 0; i < v.size(); i++) {
    printf("%d ", v[i]);
  }
  printf("\n");
}

void GraspDetector::printStdVector(const std::vector<double>& v,
                                   const std::string& name) const {
  printf("%s: ", name.c_str());
  for (int i = 0; i < v.size(); i++) {
    printf("%3.2f ", v[i]);
  }
  printf("\n");
}

}  // namespace gpd
