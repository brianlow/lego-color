import os

dataset_name = "lego-color-12-all-colors-all-cameras"


os.chdir('datasets')
os.system(f'zip -r {dataset_name}.zip {dataset_name}')
os.chdir('..')

# Save to AWS
os.system(f'aws s3 cp datasets/{dataset_name}.zip s3://brian-lego-public/lego-color/')
