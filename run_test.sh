python detect.py \
	--weights weights/ff-design2-develop/weights/best.pt \
	--conf-thres 0.65 \
	--iou-thres 0.55 \
	--save-crop \
	--save-txt \
	--save-conf \
	--img 288 \
	--name 'exp'
