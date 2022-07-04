DATA_DIR="/home/kaushikdas/nrupatunga/.exp/datasets/yolov5/fractionfish-sim/data.yaml"
python val.py \
	--img 288 \
	--conf-thres 0.65 \
	--iou-thres 0.5 \
	--data $DATA_DIR \
	--weights runs/train/fractionfish-sim/weights/best.pt \
