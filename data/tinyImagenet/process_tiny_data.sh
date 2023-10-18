wget -c http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d ./
if [ ! -d ../tinyImagenet  ];then
  mkdir ../tinyImagenet
else
  echo '../tinyImagenet' dir exist
fi

mv ./tiny-imagenet-200 ../tinyImagenet
echo move 'tiny-imagenet-200' dir to '../tinyImagenet/tiny-imagenet-200'
python tinyimagenet_reformat.py