---
layout: post
title: 윈도우환경에서 wsl ubuntu에 도커 설치하기
---

(설치하고 기억나는 대로 기록한거라서 그대로 따라한다고 될 보장없음)

# WSL Ubuntu 상에서 도커 설치하기

### **windows 기능 켜기/끄기**

Windows Subsystem for Linux(WSL) 기능을 활성화

![img1 daumcdn](https://github.com/hypro2/hypro2.github.io/assets/84513149/951bfc2d-0051-4372-9b3b-db26fe345bf5)


### **Microsoft Store에서 Ubuntu 설치**

![img1 daumcdn](https://github.com/hypro2/hypro2.github.io/assets/84513149/35cefcca-cff2-4f75-a021-6a4ff16d4573)


### **WSL 전용 NVIDIA 그래픽 드라이버 설치하기**

[https://developer.nvidia.com/cuda/wsl](https://developer.nvidia.com/cuda/wsl)

### **Ubuntu C드라이브에서 D드라이브로 옮기기**

윈도우파워셀 실행  
TAR 압축으로 Export 하기  
wsl --export Ubuntu-22.04 D:\\ubuntu-22-04.tar

기존 Ubuntu 버전 삭제  
wsl --unregister Ubuntu-22.04

압축된 TAR Import  
wsl --import Ubuntu-22.04 D:\\wsl\\ubuntu-22-04\\ D:\\ubuntu-22-04.tar

### **Ubuntu passwd 초기화**

윈도우파워셀 실행  
wsl -u root  
passwd  
passwd <사용자 계정명>

### **Ubuntu에 Docker 설치**

```
sudo apt-get update

sudo apt-get install -y \
apt-transport-https \
ca-certificates \
curl \
gnupg-agent \
software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) \
stable"

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

sudo service docker start
sudo docker run hello-world​
```

### **NVIDIA-DOCKER 설치**

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update

sudo apt-get install -y nvidia-container-toolkit

sudo service docker restart

sudo docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi

docker pull halluciation/tensorflow-gpu:v3​
```

# 컨테이너 설정 및 실행

알잘딱깔센

# VS Code에서 Container 실행하기

확장 프로그램 Dev Containers 설치

![img1 daumcdn](https://github.com/hypro2/hypro2.github.io/assets/84513149/5762769f-2934-41dc-bfdc-41e313d7eded)


컨트롤 + 시프트 + P 실행

Attach to Running Container 클릭

![img1 daumcdn](https://github.com/hypro2/hypro2.github.io/assets/84513149/dcdad8dd-9306-4c78-8241-21e729cc5bd2)


실행할 컨테이너 선택

**import module을 위한 설정**

![img1 daumcdn](https://github.com/hypro2/hypro2.github.io/assets/84513149/32ca0d7e-b670-4146-8b88-b70cbc8bb9ba)


컨트롤 + 시프트 + P 실행

Open User Settings (JSON) 클릭

**실행**

참고자료

[https://velog.io/@inthecode/Windows-10WSL%EC%97%90%EC%84%9C-Docker%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%98%EC%97%AC-Tensorflow-GPU%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0](https://velog.io/@inthecode/Windows-10WSL%EC%97%90%EC%84%9C-Docker%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%98%EC%97%AC-Tensorflow-GPU%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0)

[https://freernd.tistory.com/entry/WSL-%EC%84%A4%EC%B9%98-%EA%B2%BD%EB%A1%9C-%EB%B3%80%EA%B2%BD-%EB%B0%A9%EB%B2%95](https://freernd.tistory.com/entry/WSL-%EC%84%A4%EC%B9%98-%EA%B2%BD%EB%A1%9C-%EB%B3%80%EA%B2%BD-%EB%B0%A9%EB%B2%95)
