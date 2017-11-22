mkdir dataset
mkdir dataset/train
mkdir dataset/eval
mkdir dataset/test
cd dataset
wget "http://cs231n.stanford.edu/coco_captioning.zip"
unzip coco_captioning.zip
rm coco_captioning.zip
cd ..
python data/build_caption_dataset.py

