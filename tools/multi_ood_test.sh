checkpoints=$1
python tools/test.py local_configs/OoDTest/fishyscapes_ls.py $checkpoints
python tools/test.py local_configs/OoDTest/fishyscapes_static.py $checkpoints
python tools/test.py local_configs/OoDTest/segmeifyoucan_anomaly.py $checkpoints
python tools/test.py local_configs/OoDTest/segmeifyoucan_obstacle.py $checkpoints
python tools/test.py local_configs/OoDTest/road_anomaly.py $checkpoints