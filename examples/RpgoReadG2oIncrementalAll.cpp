/*
Example file to perform robust optimization on g2o files but incrementally
author: Yun Chang
*/

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/dataset.h>
#include <stdlib.h>

#include <fstream>
#include <memory>
#include <string>

#include "KimeraRPGO/Logger.h"
#include "KimeraRPGO/RobustSolver.h"
#include "KimeraRPGO/SolverParams.h"
#include "KimeraRPGO/utils/GeometryUtils.h"
#include "KimeraRPGO/utils/TypeUtils.h"

using namespace KimeraRPGO;

static const size_t kOptimizationBatchSize = 1;
static const size_t initTimestamp = 10000000000;
static const size_t dTimestamp = 10000000000;

template <class T>
const gtsam::SharedNoiseModel& getInitNoiseModel();

template <>
const gtsam::SharedNoiseModel& getInitNoiseModel<gtsam::Pose2>() {
  static const gtsam::SharedNoiseModel kInitNoisePose2 =
      gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(3) << 1e-6, 1e-6, 1e-6).finished());
  return kInitNoisePose2;
}

template <>
const gtsam::SharedNoiseModel& getInitNoiseModel<gtsam::Pose3>() {
  static const gtsam::SharedNoiseModel kInitNoisePose3 =
      gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished());
  return kInitNoisePose3;
}

/* Helper function to write a single Pose to CSV (template for Pose2/Pose3) */
template <class T>
void writePoseToCSV(std::ofstream& file, size_t timestamp, const T& pose);

/* Helper function to normalize rotation for Pose3 and no-op for Pose2 */
template <class T>
T normalizePoseRotationIfNeeded(const T& pose);

template <>
gtsam::Pose3 normalizePoseRotationIfNeeded<gtsam::Pose3>(
    const gtsam::Pose3& pose) {
  auto quat = pose.rotation().toQuaternion();
  quat.normalize();
  return gtsam::Pose3(gtsam::Rot3::Quaternion(quat.w(), quat.x(), quat.y(), quat.z()),
                      pose.translation());
}

template <>
gtsam::Pose2 normalizePoseRotationIfNeeded<gtsam::Pose2>(
    const gtsam::Pose2& pose) {
  return pose;
}

/* Specialization for Pose3 */
template <>
void writePoseToCSV<gtsam::Pose3>(std::ofstream& file, size_t timestamp, const gtsam::Pose3& pose) {
  gtsam::Point3 position = pose.translation();
  auto quat = pose.rotation().toQuaternion();
  file << timestamp << "," << position.x() << "," << position.y() << "," << position.z()
       << "," << quat.w() << "," << quat.x() << "," << quat.y() << "," << quat.z() << "\n";
}

/* Specialization for Pose2 */
template <>
void writePoseToCSV<gtsam::Pose2>(std::ofstream& file, size_t timestamp, const gtsam::Pose2& pose) {
  gtsam::Point2 position = pose.translation();
  double theta = pose.theta();
  file << timestamp << "," << position.x() << "," << position.y() << "," << theta << "\n";
}

/* Helper function to write all poses to CSV (template for Pose2/Pose3) */
template <class T>
void writeAllPosesToCSV(const std::string& filename, const gtsam::Values& values);

/* Specialization for Pose3 */
template <>
void writeAllPosesToCSV<gtsam::Pose3>(const std::string& filename, const gtsam::Values& values) {
  std::ofstream output(filename);
  output << "#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []\n";

  size_t timestamp = initTimestamp;
  for (size_t i = 0; i < values.keys().size(); i++) {
    gtsam::Key key = values.keys()[i];
    try {
      const gtsam::Pose3& pose = values.at<gtsam::Pose3>(key);
      writePoseToCSV<gtsam::Pose3>(output, timestamp, pose);
      timestamp += dTimestamp;
    } catch (const std::exception& e) {
      // Skip non-Pose3 values
    }
  }
  output.close();
}

/* Specialization for Pose2 */
template <>
void writeAllPosesToCSV<gtsam::Pose2>(const std::string& filename, const gtsam::Values& values) {
  std::ofstream output(filename);
  output << "#timestamp,p_RS_R_x [m],p_RS_R_y [m],theta [rad]\n";

  size_t timestamp = initTimestamp;
  for (size_t i = 0; i < values.keys().size(); i++) {
    gtsam::Key key = values.keys()[i];
    try {
      const gtsam::Pose2& pose = values.at<gtsam::Pose2>(key);
      writePoseToCSV<gtsam::Pose2>(output, timestamp, pose);
      timestamp += dTimestamp;
    } catch (const std::exception& e) {
      // Skip non-Pose2 values
    }
  }
  output.close();
}

/* Helper function to write a single Pose3 to CSV */
void writePose3ToCSV(std::ofstream& file, size_t timestamp, const gtsam::Pose3& pose) {
  gtsam::Point3 position = pose.translation();
  auto quat = pose.rotation().toQuaternion();
  file << timestamp << "," << position.x() << "," << position.y() << "," << position.z()
       << "," << quat.w() << "," << quat.x() << "," << quat.y() << "," << quat.z() << "\n";
}

/* Helper function to write all Pose3 values to CSV */
void writeAllPoses3ToCSV(const std::string& filename, const gtsam::Values& values) {
  std::ofstream output(filename);
  output << "#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []\n";

  size_t timestamp = initTimestamp;
  for (size_t i = 0; i < values.keys().size(); i++) {
    gtsam::Key key = values.keys()[i];
    try {
      const gtsam::Pose3& pose = values.at<gtsam::Pose3>(key);
      writePose3ToCSV(output, timestamp, pose);
      timestamp += dTimestamp;
    } catch (const std::exception& e) {
      // Skip non-Pose3 values
    }
  }
  output.close();
}

/* Usage: ./RpgoReadG2oIncremental <2d or 3d> <g2o file> <0 or 1 (incremental)>
    <0 - 1 (gnc probability)> <0 or 1 (multirobot frame alignment)> <translation
   threshold>
    <rotation threshold> (-1 to disable) <opt: output_folder> <opt: v for
   messages") */
template <class T>
void SimulateIncremental(gtsam::GraphAndValues gv,
                         RobustSolverParams params,
                         std::string output_folder) {
  gtsam::NonlinearFactorGraph nfg = *gv.first;
  gtsam::Values values = *gv.second;

  std::unique_ptr<RobustSolver> pgo =
      KimeraRPGO::make_unique<RobustSolver>(params);
    const gtsam::SharedNoiseModel& init_noise = getInitNoiseModel<T>();

  gtsam::Key current_key = nfg[0]->front();

  gtsam::Values init_values;  // add first value with prior factor
  gtsam::NonlinearFactorGraph init_factors;
  init_values.insert(current_key, values.at<T>(current_key));
  gtsam::PriorFactor<T> prior_factor(
      current_key, values.at<T>(current_key), init_noise);
  init_factors.add(prior_factor);

  // separate to non loop closures and loop closure factors
  gtsam::NonlinearFactorGraph non_lc_factors, lc_factors;
  for (const auto& factor : nfg) {
    if (factor_is_underlying_type<gtsam::BetweenFactor<T>>(factor)) {
      // specifically what outlier rejection handles
      gtsam::Key from_key = factor->front();
      gtsam::Key to_key = factor->back();
      if (from_key + 1 == to_key) {
        non_lc_factors.add(factor);  // odometry
      } else {
        lc_factors.add(factor);  // loop closure
      }
    } else {
      non_lc_factors.add(factor);  // not between so not lc
    }
  }
  // Add non lc factors one by one, checking for applicable loop closures
  std::vector<bool> lc_used(lc_factors.size(), false);
  size_t lc_added = 0;

  // Initialize incremental trajectory CSV
  std::string incremental_csv = output_folder + "/incremental_trajectory.csv";
  std::ofstream incremental_file(incremental_csv);
  incremental_file << "#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []\n";

  size_t timestamp = initTimestamp;

  // Write initial pose
  const T& init_pose = values.at<T>(current_key);
  writePoseToCSV(incremental_file, timestamp, init_pose);
  timestamp += dTimestamp;
  incremental_file.flush();

  // Bootstrap the solver with the first node and a prior.
  pgo->update(init_factors, init_values);

  for (size_t i = 0; i < non_lc_factors.size(); ++i) {
    const auto& non_lc_factor = non_lc_factors[i];
    gtsam::NonlinearFactorGraph new_nonlc_factors;
    new_nonlc_factors.add(non_lc_factor);
    gtsam::Values new_nonlc_values;

    // Get the max key from the non-LC factor
    gtsam::Key non_lc_max_key = std::max(non_lc_factor->front(), non_lc_factor->back());

    size_t lc_added_this_iter = 0;
    gtsam::NonlinearFactorGraph new_lc_factors;

    // Check all remaining LC factors for those to add
    for (size_t j = 0; j < lc_factors.size(); ++j) {
      if (lc_used[j]) continue;

      const auto& lc_factor = lc_factors[j];
      gtsam::Key lc_from = lc_factor->front();
      gtsam::Key lc_to = lc_factor->back();

      // Add if both indices are <= the current non-LC max key
      if (lc_from <= non_lc_max_key && lc_to <= non_lc_max_key) {
        new_lc_factors.add(lc_factor);
        lc_used[j] = true;
        ++lc_added_this_iter;
      }
    }

    lc_added += lc_added_this_iter;

    // Optimize every configured batch of loop closures, and also on the final remaining batch.
    bool optimize_graph = (lc_added_this_iter > 0 && lc_added_this_iter % kOptimizationBatchSize == 0) ||
                          (i + 1 == non_lc_factors.size());

    // Update with this non-LC and any matching LCs
    if (factor_is_underlying_type<gtsam::BetweenFactor<T>>(non_lc_factor)) {
      auto odom_factor =
          factor_pointer_cast<gtsam::BetweenFactor<T>>(non_lc_factor);
      const gtsam::Key from_key = odom_factor->front();
      const gtsam::Key to_key = odom_factor->back();

      gtsam::Values current_estimates = pgo->calculateEstimate();
      if (current_estimates.exists(from_key) && !current_estimates.exists(to_key)) {
        const T& last_estimated_pose = current_estimates.at<T>(from_key);
        const T initialized_pose =
            last_estimated_pose.compose(odom_factor->measured());
        const T normalized_pose =
            normalizePoseRotationIfNeeded<T>(initialized_pose);
        new_nonlc_values.insert(to_key, normalized_pose);
      }
    }

    pgo->update(new_nonlc_factors, new_nonlc_values);
    if (new_lc_factors.size() > 0) {
      pgo->update(new_lc_factors, gtsam::Values(), optimize_graph);
    }

    // Write the latest pose
    gtsam::Values current_estimates = pgo->calculateEstimate();
    try {
      const T& most_recent_pose = current_estimates.at<T>(non_lc_max_key);
      writePoseToCSV(incremental_file, timestamp, most_recent_pose);
      timestamp += dTimestamp;
      incremental_file.flush();
    } catch (const std::exception& e) {
      // Skip if pose cannot be extracted
    }
  }

  incremental_file.close();

  pgo->saveData(output_folder);  // tell pgo to save g2o result

  // Write final trajectory to batch CSV
  std::string batch_csv = output_folder + "/batch_trajectory.csv";
  gtsam::Values final_estimates = pgo->calculateEstimate();
  writeAllPosesToCSV<T>(batch_csv, final_estimates);
}

void PrintInputWarning(std::string err_str) {
  log<WARNING>(err_str);
  log<WARNING>(
      "Input format should be ./RpgoReadG2oIncremental <2d or 3d> <g2o file> "
      "<0 or 1 (incremental)> <0 to 1 (gnc probability, 0 or 1 to disable)> <0 "
      "or 1 (multirobot frame alignment)> <PCM trans thresh (-1 to disable)> "
      "<PCM rot thresh (-1 to disable)> <opt: output_folder> <opt: v for "
      "messages");
  log<WARNING>("Exiting application!");
}

int main(int argc, char* argv[]) {
  gtsam::GraphAndValues graphNValues;

  // A minimum of 7 arguments are required for this script to execute properly.
  // Exit early if this is the case and throw appropriate message to user.
  if (argc < 8) {
    PrintInputWarning("Missing mandatory input arguments!");
    return 0;
  }

  // Reading args and checking for validity
  bool valid_input = true;
  std::string dim = argv[1];
  int incremental = 0;
  try {
    incremental = std::stoi(argv[3]);
    if (incremental != 0 && incremental != 1) {
      throw std::invalid_argument("invalid value");
    }
  } catch (const std::invalid_argument& e) {
    std::cerr << "\"incremental\" value should be 0 or 1. You entered: "
              << argv[3] << std::endl;
    valid_input = false;
  }

  double gnc = 0;
  try {
    gnc = std::stod(argv[4]);
    if (gnc < 0 || gnc > 1) {
      throw std::invalid_argument("invalid value");
    }
  } catch (const std::invalid_argument& e) {
    std::cerr << "\"gnc\" value should be double 0 or 1. You entered: "
              << argv[4] << std::endl;
    valid_input = false;
  }

  int frame_align = 0;
  try {
    frame_align = std::stoi(argv[5]);
    if (frame_align != 0 && frame_align != 1) {
      throw std::invalid_argument("invalid value");
    }
  } catch (const std::invalid_argument& e) {
    std::cerr << "\"frame_align\" value should be 0 or 1. You entered: "
              << argv[5] << std::endl;
    valid_input = false;
  }

  double translation_t = 0.0;
  try {
    translation_t = std::stof(argv[6]);
    if (translation_t == -1 && incremental == 1) {
      log<WARNING>()
          << "incremntal mode cuurently does not support disabling pcm "
             "threshold. for now, please set to large value instead of -1";
      valid_input = false;
    }
  } catch (const std::invalid_argument& e) {
    std::cerr
        << "\"translation threshold\" value should be a float. You entered: "
        << argv[6] << std::endl;
    valid_input = false;
  }

  double rotation_t = 0.0;
  try {
    rotation_t = std::stof(argv[7]);
    if (rotation_t == -1 && incremental == 1) {
      log<WARNING>()
          << "incremntal mode cuurently does not support disabling pcm "
             "threshold. for now, please set to large value instead of -1";
      valid_input = false;
    }
  } catch (const std::invalid_argument& e) {
    std::cerr << "\"rotation threshold\" value should be a float. You entered: "
              << argv[7] << std::endl;
    valid_input = false;
  }

  // Exit application if input is invalid
  if (!valid_input) {
    PrintInputWarning("");
    return 0;
  }

  std::string output_folder;
  if (argc > 8) {
    output_folder = argv[8];
  } else {
    // saves output to current folder if not specified by user
    std::cout << "Setting output directory to current directory" << std::endl;
    output_folder = ".";
  }

  bool verbose = false;
  if (argc > 9) {
    std::string flag = argv[9];
    if (flag == "v") verbose = true;
  }
  RobustSolverParams params;

  params.logOutput(output_folder);

  if (incremental == 1) {
    params.setIncremental();
  }

  if (gnc > 0 && gnc < 1) {
    params.setGncInlierCostThresholdsAtProbability(gnc);
  }

  if (frame_align == 1) {
    params.setMultiRobotAlignMethod(MultiRobotAlignMethod::GNC);
  }

  Verbosity verbosity = Verbosity::VERBOSE;
  if (!verbose) verbosity = Verbosity::QUIET;

  if (dim == "2d") {
    graphNValues = gtsam::load2D(argv[2],
                                 gtsam::SharedNoiseModel(),
                                 0,
                                 false,
                                 true,
                                 gtsam::NoiseFormatG2O);

    params.setPcmSimple2DParams(translation_t, rotation_t, verbosity);

    SimulateIncremental<gtsam::Pose2>(graphNValues, params, output_folder);

  } else if (dim == "3d") {
    graphNValues = gtsam::load3D(argv[2]);

    params.setPcmSimple3DParams(translation_t, rotation_t, verbosity);

    SimulateIncremental<gtsam::Pose3>(graphNValues, params, output_folder);

  } else {
    PrintInputWarning("Unsupported dimension entered!");
  }
}
