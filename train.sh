 python train.py \
	 --img 288 \
	 --batch 160 \
	 --epochs 550 \
	 --data data/simreal-7-withangle/data.yml \
	 --name fractionfish \
	 --cfg models/yolov5nff.yaml \
	 --hyp data/hyps/hyp.scratch.ff.yaml
	  --weights yolov5n.pt \
