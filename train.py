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

experiment_name = "color-01-tiny"

comet_ml.init(project_name=experiment_name)

# Determine where the data will be stored. Either
#  ./data   - when running locally
#  /storage - when running on Paperspace
is_paperspace = os.environ.get('PAPERSPACE_CLUSTER_ID') is not None
data_dir = Path('/storage' if is_paperspace else './data')
if not os.path.exists(data_dir):
  os.makedirs(data_dir)

# Load a model
model = YOLO('yolov8l-cls.pt')

# Train the model
# Use an absolute path to the dataset folder. Otherwise it look
# for a folder relative to a `yolo setting` folder elsewhere
model.train(data=str(Path("./data/dataset").resolve()), name=experiment_name, epochs=10, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
