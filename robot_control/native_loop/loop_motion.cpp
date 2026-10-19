// Private socketpair controlled arm executor. No listening network endpoint.
// Production startup verifies the same evidence bundle as the Python adapter.
#include <rclcpp/rclcpp.hpp>
#include <marvin_msgs/msg/jointcmd_arm.hpp>
#include <marvin_msgs/msg/jointfeedback.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/int16_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <nlohmann/json.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <filesystem>
#include <mutex>
#include <set>
#include <sstream>
#include <map>
#include <thread>
#include <vector>
#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <dlfcn.h>
#include <cstdint>
#include <sys/syscall.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
using Json=nlohmann::json;
using Q=std::array<double,14>;
using Steady=std::chrono::steady_clock;
static volatile std::sig_atomic_t signalled=0;
static double mono(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static double wall(){timespec t;clock_gettime(CLOCK_REALTIME,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void signal_handler(int){signalled=1;}
// Use the existing system broker from this process itself. No root helper,
// capability changes, or cross-process scheduling request is involved.
struct sd_bus;
struct sd_bus_message;
struct sd_bus_error;
static int request_realtime(uint64_t tid){
  void *library=dlopen("libsystemd.so.0",RTLD_NOW|RTLD_LOCAL);if(!library)return -2;
  auto open_bus=reinterpret_cast<int(*)(sd_bus**)>(dlsym(library,"sd_bus_open_system"));
  auto call=reinterpret_cast<int(*)(sd_bus*,const char*,const char*,const char*,const char*,sd_bus_error*,sd_bus_message**,const char*,...)>(dlsym(library,"sd_bus_call_method"));
  auto timeout=reinterpret_cast<int(*)(sd_bus*,uint64_t)>(dlsym(library,"sd_bus_set_method_call_timeout"));
  auto unref=reinterpret_cast<sd_bus*(*)(sd_bus*)>(dlsym(library,"sd_bus_unref"));
  if(!open_bus||!call||!timeout||!unref){dlclose(library);return -38;}
  sd_bus *bus=nullptr;int result=open_bus(&bus);
  if(result>=0){timeout(bus,2000000);result=call(bus,"org.freedesktop.RealtimeKit1","/org/freedesktop/RealtimeKit1","org.freedesktop.RealtimeKit1","MakeThreadRealtime",nullptr,nullptr,"tu",tid,uint32_t(10));}
  if(bus)unref(bus);dlclose(library);return result;
}

static Q vector14(const Json &j){
  if(!j.is_array()||j.size()!=14)throw std::runtime_error("expected_14_joints");
  Q q; for(int i=0;i<14;i++){q[i]=j[i].get<double>();if(!std::isfinite(q[i]))throw std::runtime_error("nonfinite_joint");}return q;
}
struct Feedback{Q q{},v{};double received=0,source=0,mode_received=0,grip_received=0;int input=-1,mode=-1;bool valid=false,grip_valid=false,graph=false;std::string boot,session,grip_state;};
struct Plan{std::vector<Q> q;std::vector<std::vector<Q>> brakes;Q seed{},v0{};std::string id;double observed=0,started=0;};
struct StatusSnapshot {Feedback feedback;Q last;std::string state,reason;size_t sent=0,count=0;double started=0,max_lateness=0,sample_time=0;bool latched=false,recording_failed=false;int cpu=-1,policy=0;long tid=0;int rtkit_result=0;};
struct Trace{double due,started,completed,observed,feedback_time;Q q,measured,measured_v;int kind,index,plan_number,submitted_mask;};
class Engine{
 public:
  Json config;bool simulated;std::string ns;double dt=.002;std::atomic<double> last_supervision{0};std::mutex lock,feedback_lock,status_lock;StatusSnapshot latest_status;
  Feedback feedback;Q last{},initial{};std::array<std::array<double,2>,14> bounds{};
  std::string state="IDLE",reason,boot,session;bool bound=false,active=false,latched=false,closing=false,recording_failed=false;
  std::shared_ptr<Plan> plan;size_t sent=0,brake_index=0;std::vector<Q> brake;
  double next_due=0,started=0,max_lateness=0;std::vector<Trace> trace;size_t trace_limit=0;
  std::set<std::string> seen;std::vector<std::string> plan_ids;std::atomic<bool> quit{false};std::thread timer,ros_thread;
  rclcpp::Node::SharedPtr node;std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor;
  rclcpp::Publisher<marvin_msgs::msg::JointcmdArm>::SharedPtr left,right;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> subscriptions;rclcpp::TimerBase::SharedPtr graph_timer;
  int test_fail_side=-1,timer_cpu=-1,timer_policy=SCHED_OTHER;long timer_tid=0;int rtkit_result=0;
  Engine(Json c,bool sim,const std::string &space):config(std::move(c)),simulated(sim),ns(space){
    if(config.at("control_period_s").get<double>()!=dt)throw std::runtime_error("native_requires_2ms");
    trace_limit=std::min<size_t>(config.at("max_log_mb").get<size_t>()*1024*1024/3200,
                               size_t(config.at("max_task_duration_s").get<double>()/dt)+5000);
    trace.reserve(trace_limit);
    if(!simulated)setup_ros();
    snapshot_status();
    timer=std::thread([this]{run();});
  }
  double setting(const char *name)const{return config.at(name).get<double>();}
  void setup_ros(){
    node=std::make_shared<rclcpp::Node>("forcevla_native_"+std::to_string(getpid()),ns,
                                     rclcpp::NodeOptions().enable_rosout(false).start_parameter_services(false));
    left=node->create_publisher<marvin_msgs::msg::JointcmdArm>(ns+"/control/user/joint_cmd_A",rclcpp::QoS(1).reliable());
    right=node->create_publisher<marvin_msgs::msg::JointcmdArm>(ns+"/control/user/joint_cmd_B",rclcpp::QoS(1).reliable());
    auto sensor=rclcpp::QoS(1).best_effort();
    subscriptions.push_back(node->create_subscription<marvin_msgs::msg::Jointfeedback>(ns+"/info/joint_feedback",sensor,[this](const marvin_msgs::msg::Jointfeedback::SharedPtr m){
      std::lock_guard<std::mutex> guard(feedback_lock);feedback.valid=m->arm_positions.size()==14&&m->arm_velocities.size()==14;
      if(feedback.valid)for(int i=0;i<14;i++){feedback.q[i]=m->arm_positions[i];feedback.v[i]=m->arm_velocities[i];feedback.valid &= std::isfinite(feedback.q[i])&&std::isfinite(feedback.v[i]);}
      feedback.received=mono();feedback.source=m->header.stamp.sec+m->header.stamp.nanosec*1e-9;
    }));
    subscriptions.push_back(node->create_subscription<std_msgs::msg::Int32>(ns+"/control/input_mode",rclcpp::QoS(1).reliable().transient_local(),[this](const std_msgs::msg::Int32::SharedPtr m){std::lock_guard<std::mutex> guard(feedback_lock);feedback.input=m->data;}));
    subscriptions.push_back(node->create_subscription<std_msgs::msg::Int16MultiArray>(ns+"/info/arm_state",sensor,[this](const std_msgs::msg::Int16MultiArray::SharedPtr m){std::lock_guard<std::mutex> guard(feedback_lock);feedback.mode=(m->data.size()>=2&&m->data[0]==m->data[1])?m->data[0]:-1;feedback.mode_received=mono();}));
    std::string tool=(ns=="/tj")?"":ns;
    subscriptions.push_back(node->create_subscription<std_msgs::msg::String>(tool+"/info/web_gripper",sensor,[this](const std_msgs::msg::String::SharedPtr m){
      std::lock_guard<std::mutex> guard(feedback_lock);
      try{auto j=Json::parse(m->data);double stamp=j.at("sample_time_ns").get<double>()/1e9;double age=wall()-stamp;
        feedback.grip_valid=j.value("healthy",false)&&j.value("reserved",false)&&age>=-.02&&age<=setting("feedback_age_s");
        for(auto &a:j.at("motor_feedback_age"))feedback.grip_valid &= a.is_number()&&std::isfinite(a.get<double>())&&a.get<double>()>=0&&a.get<double>()+age<=setting("feedback_age_s");
        feedback.boot=j.value("boot_id","");feedback.session=j.at("session_id").get<std::string>();feedback.grip_state=j.value("state","");
      }catch(...){feedback.grip_valid=false;}feedback.grip_received=mono();
    }));
    graph_timer=node->create_wall_timer(std::chrono::milliseconds(50),[this]{
      bool good=left->get_subscription_count()>0&&right->get_subscription_count()>0;
      for(const auto &topic:{ns+"/control/user/joint_cmd_A",ns+"/control/user/joint_cmd_B"}){
        auto pubs=node->get_publishers_info_by_topic(topic);good &= pubs.size()==1;
        for(auto &p:pubs)good &= p.node_name()==node->get_name()&&p.node_namespace()==ns;
      }
      std::lock_guard<std::mutex> guard(feedback_lock);feedback.graph=good;
    });
    executor=std::make_shared<rclcpp::executors::SingleThreadedExecutor>();executor->add_node(node);
    ros_thread=std::thread([this]{executor->spin();});
  }
  Feedback sample(){std::lock_guard<std::mutex> guard(feedback_lock);return feedback;}
  void refresh_sim(){if(!simulated)return;std::lock_guard<std::mutex> guard(feedback_lock);feedback.q=last;feedback.v.fill(0);feedback.received=mono();feedback.source=wall();feedback.mode_received=mono();feedback.grip_received=mono();feedback.valid=feedback.grip_valid=feedback.graph=true;feedback.input=active?3:0;feedback.mode=1;feedback.boot=boot;feedback.session=session;feedback.grip_state="active";}
  std::string feedback_error(const Feedback &f,double now,bool allow_held=false){
    if(!f.valid)return "native_joint_feedback_invalid";
    if(f.input!=3)return "native_input_changed";
    if(f.mode!=1)return "native_mode_changed";
    if(!f.graph)return "native_source_graph_changed";
    if(now-f.received<0||now-f.received>setting("feedback_age_s"))return "native_joint_feedback_stale";
    if(wall()-f.source<-.02||wall()-f.source>setting("feedback_age_s"))return "native_joint_source_stale";
    if(now-f.mode_received>setting("feedback_age_s"))return "native_mode_stale";
    if(!f.grip_valid)return "native_gripper_feedback_invalid";
    if(now-f.grip_received>setting("feedback_age_s"))return "native_gripper_feedback_stale";
    if(f.boot!=boot||f.session!=session)return "native_gripper_owner_changed";
    if(f.grip_state!="active"&&!(allow_held&&f.grip_state=="held"))return "native_gripper_not_active";
    return "";
  }
  bool healthy(const Feedback &f,double now,bool allow_held=false){return feedback_error(f,now,allow_held).empty();}
  void validate_path(const std::vector<Q> &rows,Q seed,Q v0,bool final_rest,const Q &origin){
    for(const auto &q:rows){for(int j=0;j<14;j++){
      double v=(q[j]-seed[j])/dt;
      if(q[j]<bounds[j][0]||q[j]>bounds[j][1]||std::abs(q[j]-origin[j])>setting("max_joint_displacement_rad")+1e-9)throw std::runtime_error("native_joint_bound");
      if(std::abs(v)>setting("max_joint_speed_rad_s")+1e-7||std::abs(v-v0[j])/dt>setting("max_joint_acceleration_rad_s2")+1e-6)throw std::runtime_error("native_derivative_bound");
      v0[j]=v;seed[j]=q[j];
    }}
    if(final_rest)for(double v:v0)if(std::abs(v)/dt>setting("max_joint_acceleration_rad_s2")+1e-6)throw std::runtime_error("native_terminal_bound");
  }
  void abort(const std::string &why,bool braking=true){
    if(latched)return;latched=true;reason=why;
    if(braking&&state=="EXECUTE"&&plan&&sent>0){brake=plan->brakes.at(sent-1);brake_index=0;state="BRAKING";next_due=mono()+dt;}
    else{state="ABORTED";plan.reset();}
  }
  void publish(const Q &q,double due,int kind,int index,double observed,const Feedback &f){
    bool journal_full=trace.size()>=trace_limit;
    if(journal_full){recording_failed=true;if(kind!=0)throw std::runtime_error("native_journal_full");}
    double begin=mono();int submitted_mask=0;
    try {
    if(!simulated){
      auto send=[&](auto pub,int offset){marvin_msgs::msg::JointcmdArm msg;msg.header.stamp=node->now();std::copy_n(q.begin()+offset,7,msg.positions.begin());pub->publish(msg);for(int j=0;j<7;j++)last[offset+j]=q[offset+j];};
      send(left,0);submitted_mask|=1;send(right,7);submitted_mask|=2;
    }else{
      if(test_fail_side==0)throw std::runtime_error("injected_left_failure");
      for(int j=0;j<7;j++)last[j]=q[j];submitted_mask|=1;
      if(test_fail_side==1)throw std::runtime_error("injected_right_failure");
      last=q;submitted_mask|=2;
    }
    } catch(...) {
      if(trace.size()<trace_limit)trace.push_back({due,begin,mono(),observed,f.received,last,f.q,f.v,kind,index,int(plan_ids.size()),submitted_mask});
      throw;
    }
    double completed=mono();
    if(!journal_full)trace.push_back({due,begin,completed,observed,f.received,last,f.q,f.v,kind,index,int(plan_ids.size()),submitted_mask});
    if(kind==1)max_lateness=std::max(max_lateness,completed-due);
    refresh_sim();
  }
  void run(){
    // Local thread settings only: no global scheduler or device changes.
    // Reserve the final 400us for the deadline rather than depending on a
    // millisecond-scale userspace wakeup. Pinning avoids migration jitter.
    timer_tid=syscall(SYS_gettid);
    rlimit rt_limit{200000,200000};setrlimit(RLIMIT_RTTIME,&rt_limit);
    prctl(PR_SET_TIMERSLACK,1UL,0UL,0UL,0UL);
    cpu_set_t allowed;CPU_ZERO(&allowed);
    if(sched_getaffinity(0,sizeof(allowed),&allowed)==0){
      auto read_cpu=[](){std::map<int,std::pair<double,double>> samples;std::ifstream file("/proc/stat");std::string line;
        while(std::getline(file,line)){std::istringstream stream(line);std::string name;stream>>name;
          if(name.size()<4||name.substr(0,3)!="cpu"||!std::isdigit(name[3]))continue;
          double total=0,idle=0,value=0;for(int i=0;i<8&&stream>>value;i++){total+=value;if(i==3||i==4)idle+=value;}
          samples[std::stoi(name.substr(3))]={total,idle};}return samples;};
      auto before=read_cpu();std::this_thread::sleep_for(std::chrono::milliseconds(150));auto after=read_cpu();
      int chosen=-1;double best=2;
      for(const auto &entry:after){int i=entry.first;if(i>=CPU_SETSIZE||!CPU_ISSET(i,&allowed)||!before.count(i))continue;
        double total=entry.second.first-before[i].first,idle=entry.second.second-before[i].second;
        double busy=total>0?1-idle/total:1;if(busy<best){best=busy;chosen=i;}}
      if(chosen>=0){cpu_set_t one;CPU_ZERO(&one);CPU_SET(chosen,&one);if(pthread_setaffinity_np(pthread_self(),sizeof(one),&one)==0)timer_cpu=chosen;}
    }
    sched_param scheduling{};scheduling.sched_priority=10;
    if(pthread_setschedparam(pthread_self(),SCHED_FIFO,&scheduling)==0)timer_policy=SCHED_FIFO;
    else{rtkit_result=request_realtime(static_cast<uint64_t>(timer_tid));pthread_getschedparam(pthread_self(),&timer_policy,&scheduling);}
    double wake=mono();bool motion_wait=false;
    while(!quit){
      {std::lock_guard<std::mutex> guard(lock);sched_param actual{};pthread_getschedparam(pthread_self(),&timer_policy,&actual);refresh_sim();double now=mono();
       if(signalled){closing=true;abort("native_termination");}
       if(active){
        auto f=sample();now=mono(); // age clock is sampled AFTER copying callback data
        auto error=feedback_error(f,now,latched);if(!error.empty()){abort(error,false);}
        else{
          try{
            if(now-last_supervision>setting("supervision_timeout_s"))abort("native_supervisor_expired");
            if(recording_failed)abort("native_journal_full",false);
            for(int j=0;j<14;j++)if(std::abs(f.q[j]-last[j])>setting("tracking_error_rad")){abort("native_tracking",false);break;}
            if(state=="EXECUTE"){
              if(now>=next_due){
                if(now-next_due>setting("max_lateness_s")){abort("native_deadline");}
                else if(now-plan->observed>setting("max_prediction_age_s")){abort("native_prediction_expired");}
                else if(sent==plan->q.size()){state="HOLDING";plan.reset();}
                else{auto p=plan;size_t index=sent;publish(p->q[index],next_due,1,int(index),p->observed,f);sent++;double lag=trace.back().completed-next_due;next_due=p->started+(sent+1)*dt;if(lag>setting("max_lateness_s"))abort("native_submission_deadline");}
              }
            }else if(state=="BRAKING"){
              if(now>=next_due){
                if(now-next_due>setting("max_lateness_s")){state="ABORTED";plan.reset();}
                else if(brake_index==brake.size()){state="ABORTED";plan.reset();}
                else{publish(brake[brake_index],next_due,2,int(brake_index),plan?plan->observed:0,f);brake_index++;next_due+=dt;}
              }
            }else if(state=="HOLDING"||state=="ABORTED")publish(last,now,0,-1,0,f);
          }catch(const std::exception &e){abort(e.what(),false);state="ABORTED";plan.reset();}
        }
       }
       motion_wait=(state=="EXECUTE"||state=="BRAKING");
       wake=motion_wait?next_due:mono()+dt;
       snapshot_status();
      }
      double sleep_until=wake-((motion_wait&&timer_policy==SCHED_OTHER)?dt:.0004);
      if(sleep_until>mono()){
        timespec deadline{static_cast<time_t>(sleep_until),static_cast<long>((sleep_until-std::floor(sleep_until))*1e9)};
        clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&deadline,nullptr);
      }
      while(mono()<wake&&!quit)std::atomic_signal_fence(std::memory_order_seq_cst);
    }
  }
  void snapshot_status(){
    StatusSnapshot value{sample(),last,state,reason,sent,trace.size(),started,max_lateness,mono(),latched,recording_failed,timer_cpu,timer_policy,timer_tid,rtkit_result};
    std::unique_lock<std::mutex> guard(status_lock,std::try_to_lock);if(guard.owns_lock())latest_status=std::move(value);
  }
  Json status(){StatusSnapshot s;{std::lock_guard<std::mutex> guard(status_lock);s=latest_status;}
    return {{"state",s.state},{"reason",s.reason},{"sent_ticks",s.sent},{"started",s.started},{"last_q",s.last},{"measured_q",s.feedback.q},{"measured_v",s.feedback.v},{"feedback_time",s.feedback.received},{"sample_time",s.sample_time},{"max_submission_lateness_s",s.max_lateness},{"trace_count",s.count},{"latched",s.latched},{"recording_failed",s.recording_failed},{"synthetic",simulated},{"timer_cpu",s.cpu},{"timer_policy",s.policy},{"timer_tid",s.tid},{"rtkit_result",s.rtkit_result},{"deadline_spin_s",.0004},{"motion_spin_s",s.policy==SCHED_OTHER?dt:.0004}};
  }
  Json current_status(){snapshot_status();return status();}
  Json command(const Json &j){
    auto op=j.at("op").get<std::string>();
    if(j.contains("supervised_at")){double t=j["supervised_at"].get<double>();if(!std::isfinite(t)||t>mono()+.001)throw std::runtime_error("invalid_supervision_clock");double before=last_supervision.load();while(t>before&&!last_supervision.compare_exchange_weak(before,t)){}}
    if(op=="status")return status();
    std::lock_guard<std::mutex> guard(lock);
    if(op=="bind"){
      if(bound||latched)throw std::runtime_error("native_bind_once");last=initial=vector14(j.at("q"));bounds=j.at("bounds").get<decltype(bounds)>();
      for(int i=0;i<14;i++)if(!std::isfinite(bounds[i][0])||!std::isfinite(bounds[i][1])||bounds[i][0]>=bounds[i][1]||last[i]<bounds[i][0]||last[i]>bounds[i][1])throw std::runtime_error("invalid_native_bounds");
      bound=true;boot=j.at("boot").get<std::string>();session=j.at("session").get<std::string>();refresh_sim();return current_status();
    }
    if(op=="activate"){if(!bound||latched)throw std::runtime_error("native_not_bound");refresh_sim();auto f=sample();if(!simulated&&!healthy(f,mono()))throw std::runtime_error("native_activation_feedback_pending");for(int i=0;i<14;i++)if(std::abs(f.q[i]-last[i])>setting("seed_error_rad")||std::abs(f.v[i])>setting("stop_speed_rad_s"))throw std::runtime_error("native_activation_seed_changed");active=true;state="HOLDING";last_supervision=mono();refresh_sim();return current_status();}
    if(op=="deactivate"){if(state=="EXECUTE"||state=="BRAKING")throw std::runtime_error("deactivate_during_motion");active=false;state="DEACTIVATED";return current_status();}
    if(op=="abort"){abort(j.value("reason","cancelled"));return current_status();}
    if(op=="close"){closing=true;if(active)abort("native_closed");return current_status();}
    if(op=="hold"){
      auto q=vector14(j.at("q"));if(state=="EXECUTE"||state=="BRAKING")throw std::runtime_error("hold_during_segment");
      for(int i=0;i<14;i++)if(std::abs(q[i]-last[i])>1e-10)throw std::runtime_error("hold_cannot_change_target");return current_status();
    }
    if(op=="arm"){
      if(!active||latched||state!="HOLDING")throw std::runtime_error("native_requires_hold");auto f=sample();if(!healthy(f,mono()))throw std::runtime_error("native_not_ready");
      auto p=std::make_shared<Plan>();p->id=j.at("id").get<std::string>();if(p->id.empty()||seen.count(p->id))throw std::runtime_error("native_replay");
      p->seed=vector14(j.at("seed"));p->v0=vector14(j.at("seed_velocity"));p->observed=j.at("observed_at").get<double>();
      size_t n=j.at("q").size();int expected=int(std::ceil(config.at("execution_horizon").get<int>()/30.0/dt-1e-9));
      if(n!=size_t(expected)||j.at("brakes").size()!=n)throw std::runtime_error("native_horizon_mismatch");
      for(auto &row:j.at("q"))p->q.push_back(vector14(row));
      for(int k=0;k<14;k++)if(std::abs(p->seed[k]-f.q[k])>setting("seed_error_rad")||std::abs(p->seed[k]-last[k])>setting("seed_error_rad")||std::abs(f.v[k])>setting("stop_speed_rad_s"))throw std::runtime_error("native_seed_changed");
      validate_path(p->q,last,p->v0,true,p->seed);
      Q previous=last;double max_brake=0;
      for(size_t i=0;i<n;i++){
        std::vector<Q> b;for(auto &row:j["brakes"][i])b.push_back(vector14(row));
        if(b.empty()||b.size()>1000)throw std::runtime_error("native_invalid_brake");Q v;for(int k=0;k<14;k++)v[k]=(p->q[i][k]-previous[k])/dt;
        validate_path(b,p->q[i],v,true,p->seed);previous=p->q[i];max_brake=std::max(max_brake,b.size()*dt);p->brakes.push_back(std::move(b));
      }
      p->started=mono()+.01;
      if(!std::isfinite(p->observed)||p->observed>mono()||p->started+n*dt+max_brake-p->observed>setting("max_prediction_age_s"))throw std::runtime_error("native_age_budget");
      if(simulated)test_fail_side=j.value("test_fail_side",-1);
      plan=p;seen.insert(p->id);plan_ids.push_back(p->id);sent=0;started=p->started;next_due=started+dt;state="EXECUTE";return current_status();
    }
    throw std::runtime_error("unknown_native_command");
  }
  void disconnect(){std::lock_guard<std::mutex> guard(lock);closing=true;if(active)abort("native_parent_disconnected");}
  void finish(const std::string &folder){
    // Allow the checked brake to finish; no autonomous restart or release.
    double end=mono()+setting("hold_timeout_s");while(mono()<end){bool done;{std::lock_guard<std::mutex> guard(lock);done=state!="BRAKING"&&state!="EXECUTE";}if(done)break;std::this_thread::sleep_for(std::chrono::milliseconds(2));}
    quit=true;timer.join();if(executor){executor->cancel();ros_thread.join();}
    std::ofstream out(folder+"/native-ticks.jsonl");for(const auto &t:trace)out<<Json{{"due",t.due},{"started",t.started},{"completed",t.completed},{"observed_at",t.observed},{"feedback_time",t.feedback_time},{"positions",t.q},{"measured",t.measured},{"measured_velocities",t.measured_v},{"kind",t.kind},{"index",t.index},{"plan_id",t.plan_number?Json(plan_ids[t.plan_number-1]):Json(nullptr)},{"submitted_mask",t.submitted_mask}}.dump()<<'\n';
    out.flush();if(!out)throw std::runtime_error("native_tick_log_write_failed");
    std::ofstream result(folder+"/native-result.json");result<<current_status().dump(2);result.flush();if(!result)throw std::runtime_error("native_result_write_failed");
  }
};
static bool transfer(int fd,void *data,size_t n,bool sending){char *p=static_cast<char*>(data);while(n){ssize_t k=sending?send(fd,p,n,MSG_NOSIGNAL):recv(fd,p,n,0);if(k<=0)return false;p+=k;n-=k;}return true;}
int main(int argc,char **argv){
 try{
  std::string profile,ns,folder,qualification;int fd=-1,domain=-1;bool sim=false;
  for(int i=1;i<argc;i++){std::string a=argv[i];if(a=="--simulate")sim=true;else{if(i+1>=argc)throw std::runtime_error("missing_argument");std::string v=argv[++i];if(a=="--fd")fd=std::stoi(v);else if(a=="--domain")domain=std::stoi(v);else if(a=="--namespace")ns=v;else if(a=="--profile")profile=v;else if(a=="--output")folder=v;else if(a=="--qualification")qualification=v;else throw std::runtime_error("unknown_argument");}}
  sockaddr_storage address{};socklen_t size=sizeof(address);if(fd<3||getsockname(fd,reinterpret_cast<sockaddr*>(&address),&size)||address.ss_family!=AF_UNIX)throw std::runtime_error("private_socketpair_required");
  ucred peer{};size=sizeof(peer);if(getsockopt(fd,SOL_SOCKET,SO_PEERCRED,&peer,&size)||peer.uid!=getuid())throw std::runtime_error("native_peer_uid");
  if(!sim){
    if(domain==24&&ns=="/tj"){
      if(qualification.empty())throw std::runtime_error("production_qualification_required");
      const auto project = std::filesystem::canonical("/proc/self/exe")
          .parent_path().parent_path().parent_path().parent_path();
      const auto python = (project / ".control-venv/bin/python").string();
      const auto profile_path = std::filesystem::absolute(profile).string();
      const auto evidence_path = std::filesystem::absolute(qualification).string();
      pid_t child=fork();
      if(child==0){
        if(chdir(project.c_str())!=0)_exit(127);
        execl(python.c_str(),"python","-m","robot_control.check_closed_loop_qualification",
              "--profile",profile_path.c_str(),"--evidence",evidence_path.c_str(),nullptr);
        _exit(127);
      }
      int status=0;
      if(child<0||waitpid(child,&status,0)<0||!WIFEXITED(status)||WEXITSTATUS(status))
        throw std::runtime_error("production_qualification_rejected");
    }else if(domain!=231||ns.rfind("/forcevla_sandbox_",0)!=0)throw std::runtime_error("invalid_ROS_endpoint");
    setenv("ROS_DOMAIN_ID",std::to_string(domain).c_str(),1);rclcpp::init(0,nullptr);
  }
  std::signal(SIGTERM,signal_handler);std::signal(SIGINT,signal_handler);
  Json config;std::ifstream(profile)>>config;Engine engine(config,sim,ns);
  bool done=false;while(!done&&!signalled){pollfd p{fd,POLLIN,0};int ready=poll(&p,1,50);if(ready<0)break;if(!ready)continue;
    uint32_t len;if(!transfer(fd,&len,4,false))break;len=ntohl(len);if(len>4*1024*1024||!len)break;std::string input(len,'\0');if(!transfer(fd,input.data(),len,false))break;
    Json response;try{auto request=Json::parse(input);auto result=engine.command(request);response={{"ok",true},{"result",result}};done=request.value("op","")=="close";}catch(const std::exception &e){response={{"ok",false},{"error",e.what()}};}
    std::string data=response.dump();uint32_t bytes=htonl(data.size());if(!transfer(fd,&bytes,4,true)||!transfer(fd,data.data(),data.size(),true))break;
  }
  engine.disconnect();engine.finish(folder);close(fd);if(!sim)rclcpp::shutdown();return 0;
 }catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}
}
