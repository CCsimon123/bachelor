# Summary of application

This application takes a folder of unzipped SAR-images (with polarization VV+VH), a numerical approximation
of the wave height in the baltic sea and a shapefile over the coastline of the baltic sea to create two separate datasets
meant for a CNN and a ANN to train on.

The program preprocesses the SAR-image, subdivides in into nxn pixel wide subimages, checks for land, checks for 
homogeneity, selects images to create an even distribution, matches the subimage with the numerical approximation to 
apply a label to the data, writes the subimages to tif files, writes the csv files.

The output is a folder of labeled subimages in tif files and a csv file where every row consists of the mean, variance 
and label of all subimages.

-----------------------------------------------------------------------------------------------

# Setup

1. Create a new virtual environment in Anaconda. In anaconda prompt:
    conda create -n py36

2. Activate that environment and install python 3.6
    activate py36
    conda install python=3.6
   
3. Install SNAP

4. Configure snappy
   1. Locate your anaconda environment in .../anaconda3/envs/py36 and copy the path
   2. Open SNAP-command line (search in windows for it)
   3. type snappy-conf {path_to_env}\python.exe {path_to_env}\Lib\
   4. Done!
   
5. Set up your virtual environment with PyCharm
   1. File -> Settings -> Project name -> Python interpreter
   2. Click the cogwheel to add a new interpreter
   3. Go to Conda Environment
   4. Select "Existing environment"
   5. Locate your python executable in the py36 env
   6. OK

6. Fix Snappy config
   1. Go into "preprocessing"
   2. Ctrl-click on snappy in the imports
   3. scroll down to "JTS" around line 360
   4. Paste the following code under the code snippet:
    
    WriteOp = jpy.get_type('org.esa.snap.core.gpf.common.WriteOp')
    System = jpy.get_type('java.lang.System')
    PrintStream = jpy.get_type('java.io.PrintStream')
    NullStream = jpy.get_type('org.apache.commons.io.output.NullOutputStream')
    Logger = jpy.get_type('java.util.logging.Logger')
    Level = jpy.get_type('java.util.logging.Level')

7. Install the following packages using anaconda prompt or pip for part 3
   1. conda install -c anaconda numpy
   2. conda install -c conda-forge lmfit
   3. Install netcdf4 with PIP, pip install netcdf4 in terminal
    
8. Download
    1. shapefile:      https://studentchalmersse-my.sharepoint.com/:u:/g/personal/alvinge_net_chalmers_se/EQoILgavBvZMp_W_R9owZPYBRTuloFlMPcs6tKNr_7Qt8A?e=1YcFpn
    2. Model data:     https://drive.google.com/file/d/1xVw1aGQtT0iSmt3NlpEqxl-4slABKKR_/view?usp=sharing

9. Set up the following directories. All paths are relative, so you should not have to change any paths
   /Project
        /data
            /csv_output          (empty, will store output for csv)
            /img_output          (empty, will store output for images)
            /model_data          (Holds model data)
            /s1_unprocessed      (Holds unzipped SAR-images)
            /shapefile           (Holds shapefile)
       /memory                   (empty, will store memory about processed files and filled buckets)
        preprocessing.py
        plot.py
        reset.py
   
10. Select options in "preprocessing". You are now ready to run the program.

-----------------------------------------------------------------------------------------------    

# Notes

To remove all old data and memory, run the reset script. Be careful as this removes all written data in both the csv
and the images for the CNN. It also resets the folder counter and the bucket counter.

To plot the distribution of the approximated wave heights in the csv file, run the plot script






