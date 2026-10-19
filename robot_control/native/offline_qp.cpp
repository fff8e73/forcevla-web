// Standalone numerical process. No ROS initialization, sockets, or hardware SDK.
#include <marvin_qp_controller/qp_controller_core.hpp>
#include <yaml-cpp/yaml.h>
#include <Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <types.h>
static_assert(sizeof(DAQPSettings)==112,"Installed QP expects the 112-byte DAQP settings layout");

int main(int argc, char **argv) {
  try {
    if (argc != 3) throw std::runtime_error("Usage: offline_qp input.json output.yaml");
    const auto input = YAML::LoadFile(argv[1]);
    auto p = input["parameters"];
    marvin_qp_controller::QpControllerCore::Options o;
    o.package_path = input["package_path"].as<std::string>();
    o.scene_file = p["scene_file"].as<std::string>();
    o.ik_tasks_file = p["ik_tasks_file"].as<std::string>();
    o.collision_pairs_file = p["collision_pairs_file"].as<std::string>();
    o.velocity_limits_file = p["velocity_limits_file"].as<std::string>();
    o.dt = 1.0 / p["control_frequency_hz"].as<double>();
    o.posture_cost=p["posture_cost"].as<double>();
    o.damping_cost=p["damping_cost"].as<double>();
    o.kinetic_cost=p["kinetic_cost"].as<double>();
    o.qp_damping=p["qp_damping"].as<double>();
    o.collision_minimum_distance=p["collision_minimum_distance"].as<double>();
    o.collision_detection_distance=p["collision_detection_distance"].as<double>();
    marvin_qp_controller::QpControllerCore core(o);
    std::vector<double> seed(core.model().nq, 0.0);
    if (input["targets"].size()<1 || input["targets"].size()>50) throw std::runtime_error("Expected 1..50 targets");
    for (const auto &entry: input["seed"]){
      const std::string name=entry.first.as<std::string>();
      if (!core.model().existJointName(name)) throw std::runtime_error("Unknown seed joint: "+name);
      const auto &joint=core.model().joints[core.model().getJointId(name)];
      if (joint.nq()!=1) throw std::runtime_error("Expected scalar joint");
      seed[joint.idx_q()]=entry.second.as<double>();
      if (!std::isfinite(seed[joint.idx_q()])) throw std::runtime_error("Nonfinite seed");
    }
    if (input["seed"].size()!=static_cast<size_t>(core.model().nq)) throw std::runtime_error("Incomplete seed");
    if (!core.sync_configuration_from_positions(seed)) throw std::runtime_error("Seed rejected");
    core.anchor_tasks_to_configuration();
    YAML::Node output;
    output["dt"]=o.dt;
    for (pinocchio::JointIndex j=1;j<core.model().njoints;++j) {
      output["joint_names"].push_back(core.model().names[j]);
      output["q_indices"].push_back(core.model().joints[j].idx_q());
    }
    double period=input["action_period_s"].as<double>();
    if (!std::isfinite(period) || std::abs(period-1.0/30)>1e-9) throw std::runtime_error("Expected 30Hz");
    int tick=0;
    for(size_t i=0;i<input["targets"].size();++i){
      for(int side=0;side<2;++side){
        auto target=input["targets"][i][side];
        Eigen::Vector3d translation(target[0].as<double>(),target[1].as<double>(),target[2].as<double>());
        if (!translation.allFinite()) throw std::runtime_error("Nonfinite target");
        Eigen::Quaterniond rotation(target[6].as<double>(),target[3].as<double>(),target[4].as<double>(),target[5].as<double>());
        if (!rotation.coeffs().allFinite() || std::abs(rotation.norm()-1)>1e-5) throw std::runtime_error("Invalid rotation");
        if (!core.set_named_task_target(side==0?"left_hand":"right_hand",pinocchio::SE3(rotation.normalized().toRotationMatrix(),translation))) throw std::runtime_error("Task target rejected");
      }
      int end=static_cast<int>(std::ceil((i+1)*period/o.dt-1e-9));
      for(;tick<end;++tick){
        auto step=core.step();
        YAML::Node row;
        row["target_index"]=i;row["tick"]=tick;row["ok"]=step.ok;
        std::vector<double> q(core.q().data(),core.q().data()+core.q().size());
        row["q"]=q;output["steps"].push_back(row);
        if (!step.ok){output["failed_tick"]=tick;std::ofstream(argv[2])<<output;return 2;}
      }
    }
    output["status"]="completed";
    std::ofstream(argv[2])<<output;
    return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}
}
