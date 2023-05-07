# Lego Color Classification Model

![lego pieces with color predictions](docs/blues.jpeg)


## Dataset

* 42 colors, including most/all of the common colors
* real photos on white printer paper
* high confidence they are tagged correctly
  * parts are from two mostly pristine sets, so most color have some unique parts
  * [10713-1 Creative Suitcase](https://rebrickable.com/sets/10713-1/creative-suitcase/#parts)
  * [11011-1 Bricks and Animals](https://rebrickable.com/sets/11011-1/bricks-and-animals/?inventory=1#parts)


## Do it yourself

```
# gather source data
# take sheet of white US letter paper
# divide into 9 cells
# assign each cell a color
# optionally label each cell with a pen
# put lego pieces matching the color into the cells, do not overlap piecs, do not overlap cell boundaries (give yourself some margin)
# take a photo, move a little ways away to minimize fisheye
# crop/rotate photo to just the sheet of paper
# put into ./src/images and include the Rebrickable Color IDs in the filename separated by dashes, left-to-right, top-to-bottom
#   e.g. whatever-you-want.1-2-3-4-5-6-7-8-9.png

# generate dataset (see intermediate output in ./tmp)
python data.py

# optionally get on Paperspace.com
cd data
zip lego-color-dataset.zip -r dataset
aws s3 cp lego-color-dataset.zip s3://brian-lego-public/lego-color/
wget https://brian-lego-public.s3.us-west-1.amazonaws.com/lego-color/lego-color-dataset.zip

# train
python train.py
```
