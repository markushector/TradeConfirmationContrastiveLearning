**Part of Master-Thesis**
***
*Description*\
Thesis about embedding trade-confirmation pairs by utilizing 
contrastive learning. 

***
*Generating Data*\
Data was not available so first part of project was to generate
Trade-Confirmation pairs as text-files. The code for this is in gen.
Generation of files is triggered by running:\
`python main.py -N <number-of-data-samples>` \
The files will be saved in the *data*-directory.

***
*Training model*\
The code for the model is in the folder named *contrastive matching*.
The model is initialised and trained by running:\
`python train.py`\
Pre-pickled files for the dataloader are in the *disc_data_files*-directory
and pre-generated data-files are in the *disc_data*-directory. These are loaded
in the train script instead of generating new ones since that takes longer.


