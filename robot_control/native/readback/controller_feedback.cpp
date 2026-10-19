// Read-only AF_PACKET telemetry. No UDP sender, SDK session or control API.
// stdout is JSONL; all file writing is left to the unprivileged shell/consumer.
#include <marvinsdk/FxRtCSDef.h>
#include <arpa/inet.h>
#include <linux/if_ether.h>
#include <linux/filter.h>
#include <linux/if_packet.h>
#include <net/if.h>
#include <sys/socket.h>
#include <unistd.h>
#include <poll.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cerrno>

static_assert(sizeof(DCSS)==1428,"Installed DCSS ABI changed");
static_assert(sizeof(RT_OUT)==312,"Installed RT_OUT ABI changed");
static_assert(offsetof(DCSS,m_Out)==760,"Installed feedback layout changed");
static_assert(offsetof(RT_OUT,m_FB_Joint_Cmd)==88,"Joint target offset changed");

static uint16_t be16(const unsigned char *p){return static_cast<uint16_t>((p[0]<<8)|p[1]);}

bool decode(const unsigned char *packet,size_t n,DCSS &data){
  if(n<42 || be16(packet+12)!=ETH_P_IP || packet[14]>>4!=4 || packet[23]!=17)return false;
  const unsigned char source[4]={6,6,7,190};
  if(std::memcmp(packet+26,source,4)!=0 || (be16(packet+20)&0x3fff))return false;
  const size_t ihl=(packet[14]&15)*4,udp=14+ihl;
  if(ihl<20 || udp+8>n || be16(packet+udp+2)!=4730)return false;
  const size_t length=be16(packet+udp+4);
  if(length!=8+2+sizeof(DCSS) || udp+length>n)return false;
  const unsigned char *payload=packet+udp+8;
  if(payload[0]!='F'||payload[1]!='X')return false;
  std::memcpy(&data,payload+2,sizeof(data));
  for(int side=0;side<2;++side){
    const auto &a=data.m_Out[side];
    if(a.m_OutFrameSerial<0||a.m_OutFrameSerial>1000000)return false;
    for(int j=0;j<7;++j){
      if(!std::isfinite(a.m_FB_Joint_Cmd[j]) || !std::isfinite(a.m_FB_Joint_Pos[j]) || !std::isfinite(a.m_FB_Joint_Vel[j]))return false;
      if(std::abs(a.m_FB_Joint_Cmd[j])>720||std::abs(a.m_FB_Joint_Pos[j])>720)return false;
    }
  }
  return true;
}

int selftest(){
  unsigned char packet[14+20+8+2+sizeof(DCSS)]={};
  packet[12]=8;packet[14]=0x45;packet[23]=17;
  packet[26]=6;packet[27]=6;packet[28]=7;packet[29]=190;
  const int u=34;packet[u+2]=4730>>8;packet[u+3]=4730&255;
  const int length=8+2+sizeof(DCSS);packet[u+4]=length>>8;packet[u+5]=length&255;
  packet[u+8]='F';packet[u+9]='X';DCSS input{};input.m_Out[0].m_FB_Joint_Cmd[0]=90;
  input.m_Out[1].m_FB_Joint_Cmd[0]=-45;std::memcpy(packet+u+10,&input,sizeof(input));DCSS actual{};
  if(!decode(packet,sizeof(packet),actual)||actual.m_Out[0].m_FB_Joint_Cmd[0]!=90)return 1;
  packet[29]=191;if(decode(packet,sizeof(packet),actual))return 2;packet[29]=190;
  packet[20]=0x20;if(decode(packet,sizeof(packet),actual))return 3;packet[20]=0;
  if(decode(packet,sizeof(packet)-1,actual))return 4;
  input.m_Out[0].m_FB_Joint_Cmd[0]=NAN;std::memcpy(packet+u+10,&input,sizeof(input));
  if(decode(packet,sizeof(packet),actual))return 5;
  std::cout<<"{\"status\":\"selftest_passed\",\"DCSS_size\":"<<sizeof(DCSS)<<",\"target_offset\":"<<offsetof(RT_OUT,m_FB_Joint_Cmd)<<"}\n";return 0;
}

int main(int argc,char **argv){
  if(argc==2 && std::string(argv[1])=="--self-test")return selftest();
  if(argc!=1){std::cerr<<"Usage: controller_feedback [--self-test]; reads eth0 source 6.6.7.190:SDK only for 120s\n";return 2;}
  int fd=socket(AF_PACKET,SOCK_RAW,htons(ETH_P_IP));
  if(fd<0){std::cerr<<"Read-only packet socket requires CAP_NET_RAW/root: "<<std::strerror(errno)<<"\n";return 3;}
  // Kernel filter limits capture to controller IPv4/UDP and destination SDK port.
  sock_filter instructions[]={
    {0x28,0,0,12},{0x15,0,8,ETH_P_IP},{0x30,0,0,23},{0x15,0,6,17},
    {0x20,0,0,26},{0x15,0,4,0x060607be},{0xb1,0,0,14},{0x48,0,0,16},
    {0x15,0,1,4730},{0x06,0,0,65535},{0x06,0,0,0}};
  sock_fprog filter{static_cast<unsigned short>(sizeof(instructions)/sizeof(instructions[0])),instructions};
  sockaddr_ll addr{};addr.sll_family=AF_PACKET;addr.sll_protocol=htons(ETH_P_IP);addr.sll_ifindex=if_nametoindex("eth0");
  if(!addr.sll_ifindex || setsockopt(fd,SOL_SOCKET,SO_ATTACH_FILTER,&filter,sizeof(filter)) || bind(fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))){std::cerr<<"Failed to attach exact telemetry filter/bind: "<<std::strerror(errno)<<"\n";close(fd);return 4;}
  std::cout<<std::setprecision(17);unsigned char packet[4096];int last[2]={-1,-1};
  using clock=std::chrono::steady_clock;auto started=clock::now(),emitted=started;
  while(std::chrono::duration<double>(clock::now()-started).count()<120){
    pollfd p{fd,POLLIN,0};if(poll(&p,1,200)<=0)continue;
    ssize_t n=recv(fd,packet,sizeof(packet),0);DCSS d{};if(n<=0 || !decode(packet,n,d))continue;
    if(d.m_Out[0].m_OutFrameSerial==last[0] || d.m_Out[1].m_OutFrameSerial==last[1])continue;
    if(std::chrono::duration<double>(clock::now()-emitted).count()<.02)continue;
    emitted=clock::now();for(int i=0;i<2;++i)last[i]=d.m_Out[i].m_OutFrameSerial;
    const double mono=std::chrono::duration<double>(emitted.time_since_epoch()).count();
    std::cout<<"{\"source\":\"passive_controller_sdk_feedback\",\"received_mono\":"<<mono<<",\"controller_ip\":\"6.6.7.190\",\"frames\":["<<last[0]<<","<<last[1]<<"],\"commands_sent\":0";
    const char *names[]={"target_rad","positions_rad","velocities_rad_s"};
    for(int field=0;field<3;++field){std::cout<<",\""<<names[field]<<"\":[";
      for(int i=0;i<14;++i){const auto &a=d.m_Out[i/7];const float *values=field==0?a.m_FB_Joint_Cmd:field==1?a.m_FB_Joint_Pos:a.m_FB_Joint_Vel;
        if(i){std::cout<<",";}
        std::cout<<values[i%7]*3.14159265358979323846/180.;}
      std::cout<<"]";}
    std::cout<<"}\n"<<std::flush;
  }
  close(fd);return 0;
}
