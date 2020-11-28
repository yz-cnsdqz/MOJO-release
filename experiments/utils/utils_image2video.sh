#!/bin/bash

motion_gen_res="$1"

if ! [[ "$motion_gen_res" == */ ]]
then
    motion_gen_res="$motion_gen_res"/
fi


for seq in ${motion_gen_res}/*
do
    echo "$seq"
    ffmpeg -n -r 15 -i "$seq"/img_%03d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p "$seq".mp4 #-n means skip existing files and do not overwrite
    # rm -rf "$seq"
done



## example
# ./utils_imgseq2video_npz.sh /mnt/hdd/scene-aware-motiongen-data/motion_gen/results-render-000/'*MotionGenerator*'/'*'/