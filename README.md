# Lego Color Classification Model



```
# generate dataset
python data.py

# optionally get on Paperspace.com
cd data
zip lego-color-dataset.zip -r dataset
aws s3 cp lego-color-dataset.zip s3://brian-lego-public/lego-color/
wget https://brian-lego-public.s3.us-west-1.amazonaws.com/lego-color/lego-color-dataset.zip
