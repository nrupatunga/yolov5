DATA_DIR="/media/nthere/work/ml/datasets/fractionfish-alpha-2/data.yaml"

python train.py \
	--img 288 \
	--batch 160 \
	--epochs 550 \
	--data $DATA_DIR \
	--name fractionfish-real \
	--cfg models/yolov5nff.yaml \
	--hyp data/hyps/hyp.scratch.yaml \
	--weights yolov5n.pt \
