This GitHub collects all code files and directories for our Final Project for Computer Vision. Here we will generate a set of frames for the given images and apply various trackers to detect the debris objects.

1) Environment/Requirements:
   
   In the easiest case, it should be enough to install the requirements to run the given Notebook, while being in the lecture provided virtual environment of 3dcv-students.
   ``` 
   pip install -r requirements.txt
   ```
   Alternatively, create a new virtual environment with the provided environment.yml file.

3) Dataset:

   The space debris data set can be found at:
   https://www.kaggle.com/datasets/sadianawar/debris-detection-dataset/data
   Please download and unpack the Folder. The folder structure should look like this:
   ``` 
   3dcv-debris-detection
   |----------debris-detection
   |               |----------train
   |               |             |----------xxx.jpg
   |               |----------train.csv
   |----------deep_sort
   |----------**.py files
   |----------demo.ipynb

   ```

5) Deep_Sort:
   
   The provided deep_sort, as seen in https://github.com/nwojke/deep_sort/tree/master
   It needs to be in the same directory as the jupyter notebook in order for the jupyter notebook to work.

6) Notebook:
   
   The Notebook can be run now. Additionally results are provided by the students.
