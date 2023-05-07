set -o errexit  # Exit on most errors
set -o pipefail # Exit on errors in piped commands
set -o nounset  # Exit on undeclared variables
IFS=$'\n\t'     # Internal Field Separator, allow spaces in filenames

dataset_name=lego-color-common-dataset

cd data
zip -r ${dataset_name}.zip ${dataset_name}

aws s3 cp ${dataset_name}.zip s3://brian-lego-public/lego-color/
