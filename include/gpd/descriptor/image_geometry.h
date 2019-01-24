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

#ifndef IMAGE_GEOMETRY_H_
#define IMAGE_GEOMETRY_H_

#include <gpd/util/config_file.h>

namespace gpd {
namespace descriptor {

/** ImageGeometry class
 *
 * \brief Store grasp image geometry.
 *
 * This class stores parameters which define the geometry of a grasp image.
 *
 */
class ImageGeometry {
 public:
  ImageGeometry();

  ImageGeometry(double outer_diameter, double depth, double height, int size,
                int num_channels);

  ImageGeometry(const std::string& filepath);

  double outer_diameter_;  ///< the width of the volume
  double depth_;           ///< the depth of the volume
  double height_;          ///< the height of the volume
  int size_;  ///< the size of the image (image is square: width = height)
  int num_channels_;  ///< the number of channels in the image
};

std::ostream& operator<<(std::ostream& stream,
                         const ImageGeometry& hand_geometry);

}  // namespace descriptor
}  // namespace gpd

#endif /* IMAGE_GEOMETRY_H_ */
