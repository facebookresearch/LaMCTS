# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

for (( c=1; c < 600; c++ ))
do
   echo "---------------------------------"
   echo $PWD
   cd "client$c"
   echo $PWD
   screen -S client -d -m srun --gres=gpu:1 --time=24:00:00 --cpus-per-task=1 python client.py
   cd ".."
   echo "$PWD"
done

