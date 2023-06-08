**Part of Master-Thesis**
***
*Description*\
Thesis about embedding trade-confirmation pairs by utilizing 
contrastive learning. \

Model uses pre-trained glove embeddings that are to large to store on github.
For the training to run they need to be downloaded: \
`https://www.kaggle.com/datasets/adityajn105/glove6b50d`\
The downloaded file should be named `glove.6B.50d.txt` and place under `contrastive_matching/glove/`.


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
Uses the generated data-files from the `data`-directory to train.

***
**


