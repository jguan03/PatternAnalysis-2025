# Test Script:

## Step 1: 

Retrieve the semantic_MRs and semantic_labels_only files off the rangpur cluster on /home/groups/comp3710/HipMRI_Study_open/semantic_MRs and semantic_labels_only using: 

scp -r sXXXXXXX@rangpur.compute.eait.uq.edu.au:/home/groups/comp3710/HipMRI_Study_open/semantic_MRs .

scp -r sXXXXXXX@rangpur.compute.eait.uq.edu.au:/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only .

## Step 2: 

Run on local: 

scp -r sXXXXXXX@moss.labs.eait.uq.edu.au:/home/students/sXXXXXXX/semantic_MRs .

scp -r sXXXXXXX@moss.labs.eait.uq.edu.au:/home/students/sXXXXXXX/semantic_MRs .

## Step 3: 

Zip the two files and upload it to Google Drive in: /content/drive/MyDrive/

## Step 4:

Mount to Google Drive in Google Collabs (Cell 1) using: 
```
from google.colab import drive
drive.mount('/content/drive')
```
## Step 5: 

Unzip files and move the data to a temporary directory (Cell 2): 

```
%%bash
DRIVE_MRS_ZIP_PATH="/content/drive/MyDrive/semantic_MRs.zip"
DRIVE_LABELS_ZIP_PATH="/content/drive/MyDrive/semantic_labels_only.zip"

LOCAL_FAST_PATH="/tmp/data_3d"

if [ ! -d "$LOCAL_FAST_PATH" ]; then
    mkdir -p "$LOCAL_FAST_PATH"
    echo "Created base directory: $LOCAL_FAST_PATH"
else
    echo "Base directory already exists: $LOCAL_FAST_PATH"
fi

unzip -q -o $DRIVE_MRS_ZIP_PATH -d $LOCAL_FAST_PATH/

unzip -q -o $DRIVE_LABELS_ZIP_PATH -d $LOCAL_FAST_PATH/
```

## Step 6: 

Upload the module.py, dataset.py, train.py, and predict.py files to each cell in Google Collabs (Cell 3 - 6).

## Step 7: 

Run in separate cell: 

```
%run module.py
%run dataset.py
%run train.py
%run predict.py
```

or:

 Since `%%writefile filename.py` is used, you can just run `train.py` then `predict.py`. 


