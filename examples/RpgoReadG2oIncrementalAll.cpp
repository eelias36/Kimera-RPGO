/*
Example file to perform robust optimization on g2o files but incrementally
author: Yun Chang
*/

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
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

/* Helper function to write a single Pose to CSV (template for Pose2/Pose3) */
template <class T>
void writePoseToCSV(std::ofstream& file, gtsam::Key key, const T& pose);

/* Specialization for Pose3 */
template <>
void writePoseToCSV<gtsam::Pose3>(std::ofstream& file, gtsam::Key key, const gtsam::Pose3& pose) {
  gtsam::Point3 position = pose.translation();
  gtsam::Quaternion quat = pose.rotation().toQuaternion();
  file << "0," << position.x() << "," << position.y() << "," << position.z() 
       << "," << quat.w() << "," << quat.x() << "," << quat.y() << "," << quat.z() << "\n";
}

/* Specialization for Pose2 */
template <>
void writePoseToCSV<gtsam::Pose2>(std::ofstream& file, gtsam::Key key, const gtsam::Pose2& pose) {
  gtsam::Point2 position = pose.translation();
  double theta = pose.theta();
  file << "0," << position.x() << "," << position.y() << "," << theta << "\n";
}

/* Helper function to write all poses to CSV (template for Pose2/Pose3) */
template <class T>
void writeAllPosesToCSV(const std::string& filename, const gtsam::Values& values);

/* Specialization for Pose3 */
template <>
void writeAllPosesToCSV<gtsam::Pose3>(const std::string& filename, const gtsam::Values& values) {
  std::ofstream output(filename);
  output << "#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []\n";
  
  for (size_t i = 0; i < values.keys().size(); i++) {
    gtsam::Key key = values.keys()[i];
    try {
      const gtsam::Pose3& pose = values.at<gtsam::Pose3>(key);
      writePoseToCSV<gtsam::Pose3>(output, key, pose);
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
  
  for (size_t i = 0; i < values.keys().size(); i++) {
    gtsam::Key key = values.keys()[i];
    try {
      const gtsam::Pose2& pose = values.at<gtsam::Pose2>(key);
      writePoseToCSV<gtsam::Pose2>(output, key, pose);
    } catch (const std::exception& e) {
      // Skip non-Pose2 values
    }
  }
  output.close();
}

/* Helper function to write a single Pose3 to CSV */
void writePose3ToCSV(std::ofstream& file, gtsam::Key key, const gtsam::Pose3& pose) {
  gtsam::Point3 position = pose.translation();
  gtsam::Quaternion quat = pose.rotation().toQuaternion();
  file << "0," << position.x() << "," << position.y() << "," << position.z() 
       << "," << quat.w() << "," << quat.x() << "," << quat.y() << "," << quat.z() << "\n";
}

/* Helper function to write all Pose3 values to CSV */
void writeAllPoses3ToCSV(const std::string& filename, const gtsam::Values& values) {
  std::ofstream output(filename);
  output << "#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []\n";
  
  for (size_t i = 0; i < values.keys().size(); i++) {
    gtsam::Key key = values.keys()[i];
    try {
      const gtsam::Pose3& pose = values.at<gtsam::Pose3>(key);
      writePose3ToCSV(output, key, pose);
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

  size_t dim = getDim<T>();

  Eigen::VectorXd noise = Eigen::VectorXd::Zero(dim);
  static const gtsam::SharedNoiseModel& init_noise =
      gtsam::noiseModel::Diagonal::Sigmas(noise);

  gtsam::Key current_key = nfg[0]->front();

  gtsam::Values init_values;  // add first value with prior factor
  gtsam::NonlinearFactorGraph init_factors;
  init_values.insert(current_key, values.at<T>(current_key));
  gtsam::PriorFactor<T> prior_factor(
      current_key, values.at<T>(current_key), init_noise);
  nfg.add(prior_factor);

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
  
  // Initialize incremental trajectory CSV
  std::string incremental_csv = output_folder + "/incremental_trajectory.csv";
  std::ofstream incremental_file(incremental_csv);
  incremental_file << "#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []\n";
  
  // Write initial pose
  const T& init_pose = values.at<T>(current_key);
  writePoseToCSV(incremental_file, current_key, init_pose);
  incremental_file.flush();
  
  for (size_t i = 0; i < non_lc_factors.size(); ++i) {
    const auto& non_lc_factor = non_lc_factors[i];
    gtsam::NonlinearFactorGraph new_factors;
    new_factors.add(non_lc_factor);
    
    // Get the max key from the non-LC factor
    gtsam::Key non_lc_max_key = std::max(non_lc_factor->front(), non_lc_factor->back());
    
    // Check all remaining LC factors for those to add
    for (size_t j = 0; j < lc_factors.size(); ++j) {
      if (lc_used[j]) continue;
      
      const auto& lc_factor = lc_factors[j];
      gtsam::Key lc_from = lc_factor->front();
      gtsam::Key lc_to = lc_factor->back();
      
      // Add if both indices are <= the current non-LC max key
      if (lc_from <= non_lc_max_key && lc_to <= non_lc_max_key) {
        new_factors.add(lc_factor);
        lc_used[j] = true;
      }
    }
    
    // Update with this non-LC and any matching LCs
    if (i == 0) {
      pgo->update(new_factors, values);
    } else {
      pgo->update(new_factors, gtsam::Values(), false);
    }
    
    // Extract most recent pose and write to incremental CSV
    gtsam::Values current_estimates = pgo->calculateEstimate();
    try {
      const T& most_recent_pose = current_estimates.at<T>(non_lc_max_key);
      writePoseToCSV(incremental_file, non_lc_max_key, most_recent_pose);
      incremental_file.flush();
    } catch (const std::exception& e) {
      // Skip if pose cannot be extracted
    }
  }
  
  incremental_file.close();
  
  pgo->saveData(output_folder);  // tell pgo to save g2o result
  
  // Write final trajectory to batch CSV
  std::string batch_csv = output_folder + "/final_trajectory.csv";
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
