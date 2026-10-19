// Offline-only JSON-lines worker: no ROS node, publisher, socket, or control SDK.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include <marvin_qp_controller/qp_controller_core.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/parsers/mjcf.hpp>
#include <pinocchio/collision/distance.hpp>
#include "types.h"

using json = nlohmann::json;
using Core = marvin_qp_controller::QpControllerCore;
using Clock = std::chrono::steady_clock;
static_assert(sizeof(DAQPSettings) == 112, "Installed QP core requires the 112-byte DAQP settings ABI");

json load_json(const std::string &path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("Cannot read configuration: " + path);
  return json::parse(stream);
}

pinocchio::SE3 pose(const json &p, size_t offset=0) {
  Eigen::Vector3d xyz;
  for (int i=0;i<3;++i) xyz[i]=p.at(offset+i).get<double>();
  auto r=Eigen::AngleAxisd(p.at(offset+5).get<double>(), Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(p.at(offset+4).get<double>(), Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(p.at(offset+3).get<double>(), Eigen::Vector3d::UnitX());
  return {r.toRotationMatrix(),xyz};
}
json vector3(const Eigen::Vector3d &p) { return {p.x(),p.y(),p.z()}; }

class Engine {
  json cfg_;
  std::unique_ptr<Core> core_;
  pinocchio::Model model_;
  std::unique_ptr<pinocchio::Data> data_;
  pinocchio::GeometryModel geometry_;
  std::unique_ptr<pinocchio::GeometryData> geometry_data_;
  std::vector<int> qidx_;
  std::vector<std::string> names_;
  std::vector<double> velocity_;
  std::vector<std::pair<std::string,std::string>> pairs_;
  pinocchio::FrameIndex base_, left_, right_;

  void kinematics(const Eigen::VectorXd &q) {
    pinocchio::forwardKinematics(model_, *data_, q);
    pinocchio::updateFramePlacements(model_, *data_);
  }
  json skeleton() const {
    json result;
    for (auto side : {std::string("L"),std::string("R")}) {
      auto label=side=="L"?"left":"right";
      result[label]=json::array();
      for (const auto &name : chain_names(side))
        result[label].push_back(vector3(data_->oMf[model_.getFrameId(name)].translation()));
    }
    return result;
  }
  std::vector<std::string> chain_names(const std::string &side) const {
    std::vector<std::string> result={"Base_"+side};
    for(int i=1;i<=7;++i) result.push_back("Link"+std::to_string(i)+"_"+side);
    result.push_back(side=="L"?"left_tool":"right_tool");
    return result;
  }
  Eigen::VectorXd canonical_to_model(const json &j) const {
    if (!j.is_array() || j.size()!=14) throw std::runtime_error("Expected 14 canonical joint positions");
    Eigen::VectorXd q(14);
    for(int i=0;i<14;++i) {
      double v=j.at(i).get<double>();
      if(!std::isfinite(v)) throw std::runtime_error("Non-finite joint position");
      q[qidx_[i]]=v;
    }
    return q;
  }
  json model_to_canonical(const Eigen::VectorXd &q) const {
    json result=json::array();for(auto i:qidx_)result.push_back(q[i]);return result;
  }
  json residual(const pinocchio::SE3 &target, pinocchio::FrameIndex frame) const {
    auto actual=data_->oMf[base_].inverse()*data_->oMf[frame];
    return {{"position_m",(actual.translation()-target.translation()).norm()},
            {"orientation_rad",Eigen::AngleAxisd(actual.rotation().transpose()*target.rotation()).angle()}};
  }
  bool converged(const json &r) const {
    return r["position_m"].get<double>()<=cfg_["position_tolerance_m"].get<double>() &&
           r["orientation_rad"].get<double>()<=cfg_["orientation_tolerance_rad"].get<double>();
  }
  double joint_margin(const Eigen::VectorXd &q) const {
    return std::min((q-model_.lowerPositionLimit).minCoeff(),(model_.upperPositionLimit-q).minCoeff());
  }
  json distances(const Eigen::VectorXd &q) {
    pinocchio::computeDistances(model_,*data_,geometry_,*geometry_data_,q);
    double best=1e6;size_t index=0;
    for(size_t i=0;i<pairs_.size();++i) {
      double value=geometry_data_->distanceResults[i].min_distance;
      if(!std::isfinite(value))throw std::runtime_error("Non-finite collision distance");
      if(value<best){best=value;index=i;}
    }
    return {{"distance_m",best},{"pair",{pairs_[index].first,pairs_[index].second}}};
  }

public:
  explicit Engine(json cfg):cfg_(std::move(cfg)) {
    Core::Options o;
    o.package_path=cfg_.at("package_path");o.scene_file=cfg_.at("model_path");
    o.ik_tasks_file=cfg_.at("tasks_path");o.collision_pairs_file=cfg_.at("collision_pairs_path");
    o.velocity_limits_file=cfg_.at("velocity_limits_path");
    o.dt=cfg_.at("solver_dt_s");o.posture_cost=cfg_.at("posture_cost");
    o.damping_cost=cfg_.at("damping_cost");o.kinetic_cost=cfg_.at("kinetic_cost");o.qp_damping=cfg_.at("qp_damping");
    o.collision_minimum_distance=cfg_.at("minimum_distance_m");o.collision_detection_distance=cfg_.at("detection_distance_m");
    core_=std::make_unique<Core>(o);model_=core_->model();
    if(model_.nq!=14 || model_.nv!=14)throw std::runtime_error("Unexpected robot model dimensions");
    data_=std::make_unique<pinocchio::Data>(model_);
    for(auto side:{"L","R"})for(int i=1;i<=7;++i) {
      std::string name="Joint"+std::to_string(i)+"_"+side;
      if(!model_.existJointName(name))throw std::runtime_error("Missing joint: "+name);
      auto id=model_.getJointId(name);
      if(model_.joints[id].nq()!=1)throw std::runtime_error("Expected revolute scalar joint");
      names_.push_back(name);qidx_.push_back(model_.joints[id].idx_q());
    }
    for(auto name:{"base_link","left_tool","right_tool"})
      if(!model_.existFrame(name))throw std::runtime_error("Missing required model frame");
    base_=model_.getFrameId("base_link");left_=model_.getFrameId("left_tool");right_=model_.getFrameId("right_tool");
    const auto path=marvin_qp_controller::resolve_pinocchio_mjcf_model_file(o.scene_file);
    pinocchio::mjcf::buildGeom(model_,path,pinocchio::COLLISION,geometry_);
    geometry_.removeAllCollisionPairs();
    auto groups=load_json(o.collision_pairs_file).at("collision_pairs");
    std::set<std::pair<size_t,size_t>> seen;
    for(const auto &group:groups)for(const auto &a:group.at(0))for(const auto &b:group.at(1)) {
      std::string na=a,nb=b;auto ia=geometry_.getGeometryId(na),ib=geometry_.getGeometryId(nb);
      if(ia>=geometry_.ngeoms || ib>=geometry_.ngeoms)throw std::runtime_error("Configured collision geometry not found");
      auto key=std::minmax(ia,ib);
      if(seen.insert(key).second){geometry_.addCollisionPair(pinocchio::CollisionPair(ia,ib));pairs_.push_back({na,nb});}
    }
    if(pairs_.empty())throw std::runtime_error("No configured collision pairs");
    geometry_data_=std::make_unique<pinocchio::GeometryData>(geometry_);
    for(auto &r:geometry_data_->distanceRequests){r.enable_signed_distance=true;r.enable_nearest_points=true;}
    auto limits=load_json(o.velocity_limits_file).at("velocity_limits");
    for(auto &name:names_)velocity_.push_back(limits.at(name).get<double>());
  }

  json metadata() const {
    json pairs=json::array();for(const auto&p:pairs_)pairs.push_back({p.first,p.second});
    return {{"solver","installed QpControllerCore / Pinocchio / DAQP 0.7.2"},{"joint_names",names_},
      {"joint_lower",model_to_canonical(model_.lowerPositionLimit)},{"joint_upper",model_to_canonical(model_.upperPositionLimit)},
      {"velocity_limits_rad_s",velocity_},{"collision_pairs",pairs},{"collision_pair_count",pairs_.size()},
      {"coverage","Configured Link4/Link5 vs base and opposite arm only; tools, fingers, table and unlisted pairs are NOT checked"},
      {"continuous_collision_check",false},{"configuration",cfg_},{"execution_verified",false}};
  }

  json inspect(const json &request) {
    auto q=canonical_to_model(request.at("seed"));kinematics(q);
    json result={{"q",model_to_canonical(q)},{"skeleton",skeleton()},{"joint_margin_rad",joint_margin(q)}, {"collision",distances(q)}};
    kinematics(q);
    for(auto pair:{std::make_pair("left",left_),std::make_pair("right",right_)}) {
      auto m=data_->oMf[base_].inverse()*data_->oMf[pair.second];Eigen::Quaterniond quat(m.rotation());
      result["tcp"][pair.first]={{"position",vector3(m.translation())},{"quaternion_xyzw",{quat.x(),quat.y(),quat.z(),quat.w()}}};
    }
    return result;
  }

  json analyze(const json &request) {
    const auto started=Clock::now();auto seconds=[&](){return std::chrono::duration<double>(Clock::now()-started).count();};
    auto seed=canonical_to_model(request.at("seed"));const auto &actions=request.at("actions");
    if(!actions.is_array() || actions.empty() || actions.size()>50)throw std::runtime_error("Expected 1..50 action rows");
    for(const auto &row:actions){if(row.size()!=14)throw std::runtime_error("Expected 14 values per action");for(auto &v:row)if(!v.is_number()||!std::isfinite(v.get<double>()))throw std::runtime_error("Non-finite or nonnumeric action");}
    json result={{"status","complete"},{"passed",false},{"execution_verified",false},{"steps",json::array()},
      {"reasons",json::array()},{"seed",inspect(request)},{"checked_steps",0},{"total_steps",actions.size()},
      {"timing_assumption","Seed-to-first target and subsequent transitions each have 1/30 s; solver iterations are NOT execution time"}};
    if(joint_margin(seed)<-1e-6){result["status"]="invalid_seed";result["reasons"].push_back("seed_joint_limit");return result;}
    if(request.contains("observed_tcp")) {
      kinematics(seed);json check={residual(pose(request["observed_tcp"][0]),left_),residual(pose(request["observed_tcp"][1]),right_)};
      result["seed_fk_residual"]=check;
      for(auto &r:check)if(r["position_m"].get<double>()>cfg_["seed_position_tolerance_m"].get<double>()||r["orientation_rad"].get<double>()>cfg_["seed_orientation_tolerance_rad"].get<double>()){
        result["status"]="invalid_seed";result["reasons"].push_back("seed_tcp_mismatch");return result;
      }
    }
    std::vector<double> initial(seed.data(),seed.data()+seed.size());
    if(!core_->sync_configuration_from_positions(initial))throw std::runtime_error("QP core rejected seed");
    core_->anchor_tasks_to_configuration();
    Eigen::VectorXd previous=seed;bool all=true;std::set<std::string> failure_codes;
    const double dt=cfg_["step_period_s"],clearance=cfg_["minimum_distance_m"];
    for(size_t k=0;k<actions.size();++k){
      if(seconds()>cfg_["max_solve_time_s"].get<double>()){
        result["status"]="budget_exceeded";failure_codes.insert("solve_time_budget");all=false;break;
      }
      auto left=pose(actions[k]),right=pose(actions[k],7);kinematics(core_->q());
      auto base=data_->oMf[base_];
      if(!core_->set_named_task_target("left_hand",base*left)||!core_->set_named_task_target("right_hand",base*right))throw std::runtime_error("Required QP task missing");
      json residuals;bool solved=false,qp_ok=true;int iterations=0;
      for(;iterations<cfg_["max_iterations_per_step"].get<int>();++iterations){
        if(!core_->q().allFinite())throw std::runtime_error("Non-finite IK iterate");
        kinematics(core_->q());residuals={residual(left,left_),residual(right,right_)};
        if(converged(residuals[0])&&converged(residuals[1])){solved=true;break;}
        if(seconds()>cfg_["max_solve_time_s"].get<double>())break;
        if(!core_->step().ok){qp_ok=false;break;}
      }
      Eigen::VectorXd q=core_->q();if(!q.allFinite())throw std::runtime_error("Non-finite IK candidate");kinematics(q);residuals={residual(left,left_),residual(right,right_)};
      solved=converged(residuals[0])&&converged(residuals[1]);
      json reasons=json::array();if(!qp_ok)reasons.push_back("qp_failed");if(!solved)reasons.push_back("ik_not_converged");
      if(joint_margin(q)<-1e-6)reasons.push_back("joint_limit");
      double jump=(q-previous).cwiseAbs().maxCoeff();if(jump>cfg_["max_joint_step_rad"].get<double>())reasons.push_back("joint_jump");
      json velocities=json::array();bool speed_ok=true;
      for(int j=0;j<14;++j){double v=std::abs(q[qidx_[j]]-previous[qidx_[j]])/dt;velocities.push_back(v);if(v>velocity_[j]+1e-6)speed_ok=false;}
      if(!speed_ok)reasons.push_back("joint_velocity");
      if(actions[k][6].get<double>()<0||actions[k][6].get<double>()>1||actions[k][13].get<double>()<0||actions[k][13].get<double>()>1)reasons.push_back("gripper_range");
      auto closest=distances(q);int samples=std::max(1,static_cast<int>(std::ceil(jump/cfg_["collision_sample_step_rad"].get<double>())));
      if(samples>cfg_["max_interpolation_samples"].get<int>()) {reasons.push_back("collision_sampling_budget");samples=0;}
      int checked_samples=0;
      for(int i=0;i<samples;++i){
        if(seconds()>cfg_["max_solve_time_s"].get<double>()){reasons.push_back("solve_time_budget");all=false;break;}
        auto d=distances(previous+(q-previous)*(static_cast<double>(i)/samples));
        ++checked_samples;
        if(d["distance_m"].get<double>()<closest["distance_m"].get<double>())closest=d;
      }
      if(closest["distance_m"].get<double>()<clearance)reasons.push_back("configured_collision_clearance");
      kinematics(q);
      result["steps"].push_back({{"index",k},{"converged",solved},{"qp_ok",qp_ok},{"iterations",iterations},{"q",model_to_canonical(q)},
        {"skeleton",skeleton()},{"residual",residuals},{"joint_margin_rad",joint_margin(q)}, {"max_joint_step_rad",jump},
        {"joint_speeds_rad_s",velocities},{"closest",closest},{"interpolation_samples",checked_samples},{"interpolation_samples_required",samples},{"reasons",reasons},{"passed",reasons.empty()}});
      for(auto &code:reasons)failure_codes.insert(code.get<std::string>());
      all=all&&reasons.empty();
      if(solved) previous=q;
      else { // A failed numerical candidate is never used as the next branch seed.
        std::vector<double> reset(previous.data(),previous.data()+previous.size());core_->sync_configuration_from_positions(reset);
      }
    }
    result["checked_steps"]=result["steps"].size();result["reasons"]=failure_codes;
    if(failure_codes.count("solve_time_budget")) result["status"]="budget_exceeded";
    result["passed"]=all && result["steps"].size()==actions.size();result["elapsed_s"]=seconds();return result;
  }
};

int main(int argc,char**argv) {
  FILE *protocol=fdopen(dup(STDOUT_FILENO),"w");dup2(STDERR_FILENO,STDOUT_FILENO);
  auto send=[&](const json &value){auto s=value.dump();std::fprintf(protocol,"%s\n",s.c_str());std::fflush(protocol);};
  try {
    if(argc!=2)throw std::runtime_error("Usage: offline_ik_worker CONFIG_JSON");
    Engine engine(load_json(argv[1]));send({{"ready",true},{"metadata",engine.metadata()}});
    std::string line;
    while(std::getline(std::cin,line)) {
      std::string request_id;
      try {
        auto request=json::parse(line);request_id=request.value("request_id","");
        json result=request.value("operation","analyze")=="inspect"?engine.inspect(request):engine.analyze(request);
        result["request_id"]=request.value("request_id","");send(result);
      } catch(const std::exception&e){send({{"status","error"},{"request_id",request_id},{"passed",false},{"error",e.what()},{"execution_verified",false}});}
    }
  }catch(const std::exception&e){send({{"ready",false},{"error",e.what()}});return 1;}
  std::fclose(protocol);return 0;
}
