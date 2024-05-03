# Tasks

1. Test on a single image
2. Detection of batch images and insertion into database
3. Main application for use

## Breakdown

### Task 0: Setting up Environment
- Open terminal
- Activating virtual environment using `conda activate denvee`
- Navigate to 'mmdetection' folder using `cd mmdetection`

### Task 1: Test on a single image
- Run the code below in terminal, where demo/girl.png is the image to be detected
- `PYTHONPATH='.' python demo/ntt.py demo/girl.png configs/fashionformer/fashionpedia/fashionformer_r101_mlvl_feat_8x.py ../data/model_final.pth --device cpu --out-file demoe.png --score-thr 0.5`
- demo/ntt.py is the script to perform the detection
- demo/girl.png is the file to be detected
- configs/fashionformer/fashionpedia/fashionformer_r101_mlvl_feat_8x.py is the config file
- ../data/model_final.pth is the model weight
The rest settings should remain
- Output image of detection will be found inside the mmdetection in demoe.png file

### Task 2: Add to database
- Run the code below in terminal, where batch is the folder that contains all the images to be added
- `PYTHONPATH='.' python demo/process_db.py batch configs/fashionformer/fashionpedia/fashionformer_r101_mlvl_feat_8x.py ../data/model_final.pth --device cpu --score-thr 0.5`
- The output images can be found in `outputx` folder, inside the `mmdetection` folder

### Task 3: Running the Main App
- Open terminal and run the following code:
- `PYTHONPATH='.'  streamlit run demo/app.py`
- Accept option to make app public
- Switch to ports tab and press the web icon to open on a browser