# Lego Color Classification Model

![lego pieces with color predictions](docs/purples.jpg)

## Dataset

* 42 colors, including most/all of the common colors
* real photos on white printer paper
* labeled with Rebrickable.com color ids
* high confidence they are tagged correctly
  * parts are from two mostly pristine sets
  * most colors have some unique parts so I am confident that are labeled correctly
  * [10713-1 Creative Suitcase](https://rebrickable.com/sets/10713-1/creative-suitcase/#parts)
  * [11011-1 Bricks and Animals](https://rebrickable.com/sets/11011-1/bricks-and-animals/?inventory=1#parts)

Stats
```
Class:  # Train   # Val   Total
------  -------   -----   -----
0    :       20       4      24
1    :      203      61     264
2    :       61      16      77
3    :      205      52     257
4    :       46      16      62
5    :       66      16      82
10   :       37       4      41
14   :      293      92     385
15   :       62      16      78
19   :       66      16      82
25   :      324      65     389
26   :       40      11      51
27   :       70      25      95
28   :      106      23     129
29   :       78      16      94
30   :       44       9      53
31   :       51      13      64
41   :      188      34     222
46   :       70      17      87
47   :       55      14      69
70   :       65      15      80
71   :       51      11      62
72   :       80      13      93
73   :      345      94     439
84   :       46      11      57
85   :       29      13      42
158  :       18       7      25
182  :       58      12      70
191  :      277      59     336
212  :      235      51     286
226  :      249      62     311
272  :      289      72     361
288  :       52      11      63
308  :       30      11      41
320  :      168      52     220
321  :      256      48     304
322  :      246      57     303
323  :      109      37     146
326  :       43      12      55
378  :       28       7      35
484  :       54      15      69
1050 :       59      10      69
------  -------   -----   -----
Total      4872    1200    6072
```

## Future

* more colors
* vary the background
* printed parts


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
python dataset.py

# optionally get on Paperspace.com
aws s3 cp datasets/lego-color-common-5k-dataset-4-baseline-plus-renders.zip s3://brian-lego-public/lego-color/
wget https://brian-lego-public.s3.us-west-1.amazonaws.com/lego-color/lego-color-common-5k-dataset-4-baseline-plus-renders.zip

# train
python train.py
```
