#!/usr/bin/env bash
# Build the KCBS-instrumented Dream receiver. Run inside WSL/Ubuntu.
set -e
cd "$(dirname "$0")"
if [ ! -d dream-src ]; then
  git clone --depth 1 https://github.com/mmjee/dream-patched.git dream-src
fi
python3 apply_kcbs_patch.py dream-src
sudo apt-get update
sudo apt-get install -y build-essential g++ make qtbase5-dev qtmultimedia5-dev \
  libqt5svg5-dev qttools5-dev-tools libqwt-qt5-dev libfftw3-dev libopus-dev \
  libsndfile1-dev libasound2-dev libpcap-dev libhamlib-dev libspeexdsp-dev \
  libfdk-aac-dev libfaad-dev zlib1g-dev pkg-config
cd dream-src
qmake dream.pro
make -j"$(nproc)"
echo
echo "built: $(ls -la dream 2>/dev/null || find . -maxdepth 2 -name dream -type f)"
