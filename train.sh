DATA_DIR="/home/kaushikdas/nrupatunga/.exp/datasets/yolov5/fractionfish-sim/data.yaml"
python train.py \
	--img 288 \
	--batch 160 \
	--epochs 550 \
	--data $DATA_DIR \
	--name fractionfish-sim \
	--cfg models/yolov5nff.yaml \
	--hyp data/hyps/hyp.scratch.ff.yaml \
	--weights yolov5n.pt \
