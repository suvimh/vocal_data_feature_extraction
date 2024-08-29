Notebook and functions used to extract audio features from multimodal vocal data.

1. Start a virtual environment in python 3.10 (can't use >3.11 because of mireval used in crepe pitch extraction) (e.g. using conda): 
conda create -n data_extract_env python=3.10
conda activate data_extract_env

2. Install requirements: 
conda install -c conda-forge cmake
pip install -r requirements.txt

3. Running the code:
Follow step by step of python notebook. If interested in the internal workings of the
functions used, look at the other python scripts containing them.

The metadata of participants used to lable the output CSV rows with metadata is not committed to this repository to protect the privacy of participants. Example data in the media folder provided is of myself. They are all of the same take of a simple triad scale sung with bad posture.



RUNNING THE SCRIPTS
If modules are not recognised, make sure that your project is in your path, 
e.g. run the following:
export PYTHONPATH="/Users/XXX/Documents/UPF/THESIS/feature_extraction:$PYTHONPATH"
export PYTHONPATH="path_to/feature_extraction:$PYTHONPATH"


RUNNING IN COLAB
The code has also been modified so that it runs within Google Colab. This version can be accessed at: 
https://drive.google.com/drive/folders/1n0oCDDvmJ6Y0GNAAFF7UNw-SH2JxeegB?usp=sharing
Note that this version of the code can be viewed. To run in Colab, all code and data needs to be within Google Drive, 
paths updated accordingly and the module imports handled differently. This is all set up in the Google Colab notebook
viewed in the link. 