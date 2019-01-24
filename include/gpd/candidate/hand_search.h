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

#ifndef HAND_SEARCH_H
#define HAND_SEARCH_H

#include <Eigen/Dense>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/random_sample.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>

#include <omp.h>

#include <memory>

#include <gpd/candidate/antipodal.h>
#include <gpd/candidate/finger_hand.h>
#include <gpd/candidate/frame_estimator.h>
#include <gpd/candidate/hand.h>
#include <gpd/candidate/hand_geometry.h>
#include <gpd/candidate/hand_set.h>
#include <gpd/candidate/local_frame.h>
#include <gpd/util/plot.h>
#include <gpd/util/point_list.h>

namespace gpd {
namespace candidate {

typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGB;

/** HandSearch class
 *
 * \brief Search for grasp hypotheses.
 *
 * This class searches for grasp hypotheses in a point cloud by first
 * calculating a local reference frame for a small
 * point neighborhood, and then finding geometrically feasible grasp hypotheses
 * for a larger point neighborhood. It
 * can also estimate whether the grasp is antipodal from the normals of the
 * point neighborhood.
 *
 */
class HandSearch {
 public:
  struct Parameters {
    /** local reference frame estimation parameters */
    double nn_radius_frames_;  ///< local reference frame radius for point
                               /// neighborhood search

    /** grasp candidate generation */
    int num_threads_;  ///< the number of CPU threads to be used for the
                       /// hypothesis generation
    int num_samples_;  ///< the number of samples drawn from the point clouds;
    Eigen::Matrix4d cam_tf_left_;   ///< pose of the left camera
    Eigen::Matrix4d cam_tf_right_;  ///< pose of the right camera
    int num_orientations_;          ///< number of hand orientations to evaluate
    std::vector<int> hand_axes_;    ///< the rotation axis about which different
                                    /// hand orientations are generated

    HandGeometry hand_geometry_;  ///< robot hand geometry
  };

  /**
   * \brief Constructor.
   * \param params Parameters for the hand search
   */
  HandSearch(Parameters params);

  /**
   * \brief Search robot hand configurations.
   * \param cloud_cam the point cloud
   * \param plots_normals if surface normals are visualized
   * \param plots_samples if samples are visualized
   * \return list of grasp candidate sets
   */
  std::vector<std::unique_ptr<candidate::HandSet>> searchHands(
      const util::Cloud& cloud_cam, bool plots_normals = false,
      bool plots_samples = false) const;

  /**
   * \brief Reevaluate a list of grasp candidates.
   * \param cloud_cam the point cloud
   * \param grasps the list of grasp candidates
   * \param plots_samples if samples are visualized
   */
  std::vector<int> reevaluateHypotheses(
      const util::Cloud& cloud_cam,
      std::vector<std::unique_ptr<candidate::Hand>>& grasps,
      bool plot_samples = false) const;

  const Parameters& getParams() const { return params_; }

  /**
   * \brief Set the parameters for the hand search.
   * \param params the parameters
   */
  void setParameters(const Parameters& params) { params_ = params; }

 private:
  /**
   * \brief Search robot hand configurations given a list of local reference
   * frames.
   * \param cloud_cam the point cloud
   * \param frames the list of local reference frames
   * \param kdtree the KDTree object used for fast neighborhood search
   * \return the list of robot hand configurations
   */
  std::vector<std::unique_ptr<candidate::HandSet>> evalHands(
      const util::Cloud& cloud_cam,
      const std::vector<candidate::LocalFrame>& frames,
      const pcl::KdTreeFLANN<pcl::PointXYZRGBA>& kdtree) const;

  /**
   * \brief Reevaluate a grasp candidate.
   * \param point_list the point neighborhood associated with the grasp
   * \param hand the grasp
   * \param finger_hand the FingerHand object that describes a valid finger
   * placement
   * \param point_list_cropped the point neigborhood transformed into the hand
   * frame
   */
  bool reevaluateHypothesis(const util::PointList& point_list,
                            const candidate::Hand& hand,
                            FingerHand& finger_hand,
                            util::PointList& point_list_cropped) const;

  /**
   * \brief Calculate the label for a grasp candidate.
   * \param point_list the point neighborhood associated with the grasp
   * \param finger_hand the FingerHand object that describes a valid finger
   * placement
   * \return the label
   */
  int labelHypothesis(const util::PointList& point_list,
                      FingerHand& finger_hand) const;

  /**
   * \brief Convert an Eigen::Vector3d object to a pcl::PointXYZRGBA.
   * \param v the Eigen vector
   * \reutrn the pcl point
   */
  pcl::PointXYZRGBA eigenVectorToPcl(const Eigen::Vector3d& v) const;

  Parameters params_;  ///< parameters for the hand search

  double nn_radius_;  ///< radius for nearest neighbors search

  Eigen::Matrix3Xd cloud_normals_;  ///< a 3xn matrix containing the normals for
                                    /// points in the point cloud
  std::unique_ptr<util::Plot> plot_;  ///< plot object for visualization

  /** plotting parameters (optional, not read in from config file) **/
  bool plots_samples_;  ///< are the samples drawn from the point cloud plotted?
  bool plots_camera_sources_;  ///< is the camera source for each point in the
                               /// point cloud plotted?
  bool plots_local_axes_;      ///< are the local axes estimated for each point
                               /// neighborhood plotted?

  /** constants for rotation axis */
  static const int
      ROTATION_AXIS_NORMAL;  ///< normal axis of local reference frame
  static const int
      ROTATION_AXIS_BINORMAL;  ///< binormal axis of local reference frame
  static const int ROTATION_AXIS_CURVATURE_AXIS;  ///< curvature axis of local
                                                  /// reference frame
};

}  // namespace candidate
}  // namespace gpd

#endif /* HAND_SEARCH_H */
