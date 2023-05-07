# Download the dataset from AWS to Paperspace
# Run this in a Paperspace Gradient terminal

set -o errexit  # Exit on most errors
set -o pipefail # Exit on errors in piped commands
set -o nounset  # Exit on undeclared variables
IFS=$'\n\t'     # Internal Field Separator, allow spaces in filenames

dataset_name=lego-color-common-dataset

cd /storage
wget https://brian-lego-public.s3.us-west-1.amazonaws.com/lego-color/${dataset_name}.zip
unzip ${dataset_name}.zip
rm ${dataset_name}.zip
