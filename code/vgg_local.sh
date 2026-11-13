#!/bin/sh
# resumable download of a shard to disk, then extract, then delete
for i in "$@"; do
  f=${EV_DATA:-data}/vgg/shard_$i.tar.gz
  url=https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_$(printf %02d $i).tar.gz
  size=$(curl -sIL $url | grep -i '^content-length' | tail -1 | tr -dc 0-9)
  n=0
  while [ "$(stat -c %s $f 2>/dev/null || echo 0)" != "$size" ] && [ $n -lt 60 ]; do
    curl -sL -C - --retry 10 -o $f $url; n=$((n+1))
  done
  echo "shard $i downloaded $(stat -c %s $f) of $size after $n tries"
  VGG_LOCAL=$f python3 "$(dirname "$0")/vgg_stream.py" $i ${EV_DATA:-data}/vgg/out ${EV_DATA:-data}/vgg/vggsound.csv
  rm -f $f
done
echo ALLDONE > ${EV_DATA:-data}/vgg/all.flag
