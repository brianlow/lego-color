import os
import comet_ml
import sys
# sys.path.insert(0, '../ultralytics') # uncomment to use local clone of repo
from ultralytics import YOLO
from pathlib import Path

#
# Yolo v8 training script
#

# Paperspace setup
# export COMET_API_KEY=<your key if you want to report to comet.com>
# pip install -r requirements.txt!

experiment_name = "color-03-common-5k"
dataset_name = "lego-color-common-5k-dataset"

comet_ml.init(project_name=experiment_name)

# Determine where the data will be stored. Either
#  ./datasets   - when running locally
#  /storage - when running on Paperspace
is_paperspace = os.environ.get('PAPERSPACE_CLUSTER_ID') is not None
data_dir = '/storage' if is_paperspace else './datasets'
data_dir = str(Path(data_dir).resolve())

# Load a model
model = YOLO('yolov8l-cls.pt')

# Train the model
# Use an absolute path to the dataset folder. Otherwise it look
# for a folder relative to a `yolo setting` folder elsewhere
model.train(data=os.path.join(data_dir, dataset_name), name=experiment_name, epochs=300, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
