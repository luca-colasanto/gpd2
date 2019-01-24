/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2018, Andreas ten Pas
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef GRASP_DETECTOR_H_
#define GRASP_DETECTOR_H_

// System
#include <algorithm>
#include <memory>
#include <vector>

// PCL
#include <pcl/common/common.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <gpd/candidate/candidates_generator.h>
#include <gpd/candidate/hand_geometry.h>
#include <gpd/candidate/hand_set.h>
#include <gpd/clustering.h>
#include <gpd/descriptor/image_generator.h>
#include <gpd/net/classifier.h>
#include <gpd/util/config_file.h>
#include <gpd/util/plot.h>

namespace gpd {

/** GraspDetector class
 *
 * \brief Detect grasp poses in point clouds.
 *
 * This class detects grasps in a point cloud by first creating a large set of
 * grasp hypotheses, and then classifying each of them as a grasp or not.
 *
 */
class GraspDetector {
 public:
  /**
   * \brief Constructor.
   * \param node ROS node handle
   */
  GraspDetector(const std::string& config_filename);

  std::vector<std::unique_ptr<candidate::Hand>> detectGrasps(
      const util::Cloud& cloud);

  /**
   * \brief Preprocess the point cloud.
   * \param cloud_cam the point cloud
   */
  void preprocessPointCloud(util::Cloud& cloud);

  void sampleAbovePlane(util::Cloud& cloud);

  std::vector<std::unique_ptr<candidate::HandSet>> filterGraspsWorkspace(
      std::vector<std::unique_ptr<candidate::HandSet>>& hand_set_list,
      const std::vector<double>& workspace) const;

  /**
   * \brief Filter side grasps that are close to the table.
   * \param hand_set_list list of grasp candidate sets
   */
  std::vector<std::unique_ptr<candidate::HandSet>> filterSideGraspsCloseToTable(
      std::vector<std::unique_ptr<candidate::HandSet>>& hand_set_list);

  /**
   * \brief Generate grasp candidates.
   * \param cloud_cam the point cloud
   * \return the list of grasp candidates
   */
  std::vector<std::unique_ptr<candidate::HandSet>> generateGraspCandidates(
      const util::Cloud& cloud);

  bool createGraspImages(
      util::Cloud& cloud,
      std::vector<std::unique_ptr<candidate::Hand>>& hands_out,
      std::vector<std::unique_ptr<cv::Mat>>& images_out);

  std::vector<int> evalGroundTruth(
      const util::Cloud& cloud_gt,
      std::vector<std::unique_ptr<candidate::Hand>>& hands);

  std::vector<std::unique_ptr<candidate::Hand>> pruneGraspCandidates(
      const util::Cloud& cloud,
      const std::vector<std::unique_ptr<candidate::HandSet>>& hand_set_list,
      double min_score);

  std::vector<std::unique_ptr<candidate::Hand>> selectGrasps(
      std::vector<std::unique_ptr<candidate::Hand>>& hands) const;

  static bool isScoreGreater(const std::unique_ptr<candidate::Hand>& hand1,
                             const std::unique_ptr<candidate::Hand>& hand2) {
    return hand1->getScore() > hand2->getScore();
  }

  const candidate::HandSearch::Parameters& getHandSearchParameters() {
    return candidates_generator_->getHandSearchParams();
  }

  const descriptor::ImageGeometry& getImageGeometry() const {
    return image_generator_->getImageGeometry();
  }

 private:
  void printStdVector(const std::vector<int>& v, const std::string& name) const;

  void printStdVector(const std::vector<double>& v,
                      const std::string& name) const;

  std::unique_ptr<candidate::CandidatesGenerator> candidates_generator_;
  std::unique_ptr<descriptor::ImageGenerator> image_generator_;
  std::unique_ptr<Clustering> clustering_;
  std::unique_ptr<util::Plot> plotter_;
  std::shared_ptr<net::Classifier> classifier_;

  // classification parameters
  double min_score_;           ///< minimum classifier confidence score
  bool create_image_batches_;  ///< if images are created in batches (reduces
                               /// memory usage)

  // plotting parameters
  bool plot_normals_;              ///< if normals are plotted
  bool plot_samples_;              ///< if samples/indices are plotted
  bool plot_filtered_candidates_;  ///< if filtered grasp candidates are plotted
  bool plot_volumes_;       ///< if volumes associated with grasps are plotted
  bool plot_valid_grasps_;  ///< if valid grasps are plotted
  bool plot_clustered_grasps_;  ///< if clustered grasps are plotted
  bool plot_selected_grasps_;   ///< if selected grasps are plotted

  // filtering parameters
  bool filter_table_side_grasps_;  ///< if side grasps close to the table are
                                   /// filtered
  bool cluster_grasps_;            ///< if grasps are clustered
  double min_aperture_;  ///< the minimum opening width of the robot hand
  double max_aperture_;  ///< the maximum opening width of the robot hand
  std::vector<double> workspace_grasps_;  ///< the workspace of the robot with
                                          /// respect to hand poses
  std::vector<double> vert_axis_;  ///< vertical axis used for filtering side
                                   /// grasps that are close to the table
  double table_height_;            ///< height of table (along vertical axis)
  double table_thresh_;  ///< distance threshold below which side grasps are
                         /// considered to be too close to the table
  double angle_thresh_;  ///< angle threshold below which grasps are considered
                         /// to be side grasps

  // selection parameters
  int num_selected_;  ///< the number of selected grasps
};

}  // namespace gpd

#endif /* GRASP_DETECTOR_H_ */
