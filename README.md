# Morphological-Inflection
Morphological Inflection in 52 languages, part of SIGMORPHON 2017 shared task

This work was done during my internship at NLP Research Lab, IIT (BHU) in 2017, in contribution to the SIGMORPHON 2017 shared task (https://sites.google.com/view/conll-sigmorphon2017/home). We took part in only the first subtask. Our paper can be found at http://www.aclweb.org/anthology/K17-2007. 

## Note

This repo contains code for one set of configurations described in the paper. We have submitted many configurations, the code for the rest of which, will be uploaded soon. This repo is still in construction but the code existing currently is complete in its own regard.

## Usage:

1) Create directories - datasets, keras_saved_models, output_files in the program directory
2) Datasets can be found at : https://github.com/sigmorphon/conll2017/tree/master/all/task1
3) Place all the files in the above dataset repo in the 'datasets' directory created
4) Run prog.py
5) Upon prog.py's running, keras models are saved in the 'keras_saved_models' directory, and output_files are generated in output_files directory
7) data.py does helper tasks of data processing on the datasets
6) eval.py is an evaluation script that can be run on the output files (look inside eval.py for how to run it)
