#####################################################################################
# Siamese Neural Networks for One-Shot Image Recognition (tracey fork + instructions)
- I added `tracey.py` as a single .py file extracted and lightly modified from the jupyter notebook
- I disabled 'cuda' since running on mac laptop


# Setup:

```bash
sudo easy_install pip;
sudo pip install --upgrade virtualenv;
virtualenv --system-site-packages -p python3 .;
source bin/activate;

# from pytorch.org install: osx pip 3.6 8.0
pip3 install http://download.pytorch.org/whl/torch-0.2.0.post3-cp36-cp36m-macosx_10_7_x86_64.whl;
pip3 install torchvision;

pip3 install matplotlib;
mkdir -p  ~/.matplotlib;
echo 'backend: TkAgg' >| ~/.matplotlib/matplotlibrc;

pip3 install -r requirements.txt;
```
- if requirements fails, comment out 2 *torch* related requirements since installed separately


# my dataset (clay statues -- pictures of torsos) -- crop to heads:
```bash
DEST=$(pwd)/data/faces/training;
mkdir -p $DEST;
cd ~/dev/train/__masked;
for i in */*.{png,jpg}; do
  CROP='556x580+300+146!';
  base=$(basename $i)
  if [ "$base" = "c.png" ]; then
    CROP='506x491+322+146!';
  fi
  mkdir -p $(dirname $DEST/$i);
  convert $i -crop "$CROP" $DEST/$i  &&  echo -n .;
done
cd -
```

# move 10 random sets from 'training' to 'testing':
```bash
cd data/faces/;
mkdir -p testing;
mv testing/* training/;
mv  $(/bin/ls -d training/* |gshuf |head -10)  testing/;
cd -;
```

#run
```bash
source bin/activate;
python3 tracey.py;
```


---
# What follows is from the forked repo:
---

# Facial Similarity with Siamese Networks in Pytorch
You can read the accompanying article at https://hackernoon.com/one-shot-learning-with-siamese-networks-in-pytorch-8ddaab10340e

The goal is to teach a siamese network to be able to distinguish pairs of images.
This project uses pytorch.

Any dataset can be used. Each class must be in its own folder. This is the same structure that PyTorch's own image folder dataset uses.

### Converting pgm files (if you decide to use the AT&T dataset) to png
1. Install imagemagick
2. Go to root directory of the images
3. Run `find -name "*pgm" | xargs -I {} convert {} {}.png`


You can find the project requirements in requirements.txt

#### This project requires python3.6
