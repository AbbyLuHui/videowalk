python -W ignore train.py --data-path /proj/vondrick/datasets/CATER/GREATER_multi_fbb/train/trainlist_multiview.txt \
--frame-aug grid --dropout 0.1 --clip-len 4 --temp 0.05 --visualize \
--model-type scratch --workers 16 --batch-size 16  \
--cache-dataset --data-parallel --lr 0.0001
