# EmEL-V
EmEL-V is a geometric approach to generate embeddings for the description logic EL++
The implementation is done using Python and Pytorch Library.

## Requirements      
Click 7.0    
cycler 0.10.0    
gast 0.2.2    
grpcio 1.18.0    
Keras-Applications 1.0.6    
Keras-Preprocessing 1.0.5    
numpy 1.16.0    
pandas 0.23.4    
pkg-resources 0.0.0         
pytz 2018.9    
scikit-learn 0.20.2    
scipy 1.2.0    
six 1.12.0     
sklearn 0.0  
torch 1.5.0

tensorboard 1.15.0     
tensorflow-gpu 1.15.0 

The code is organized as follows:
- Experiments: This contains separate folder for each ontology the experiment is carried out upon.
- Experiments folder contains models, data and results folder(create an empty results folder and the others required to store the model)
- models folder contains code which takes in the dataset names
- The corresponding dataset folders must be present in the data folder
- Corresponding results folder stores the trained model parameters(make sure to change the path for out_file in the code)
- The implementation of evaluation metrics - Evaluating_HITS.py

-experiments/data/{dataset_name} : This folder consists of 4 processed files namely, normalized form of the ontology file
to be used for training, and training,validation & testing set obtained from subclass relations in ontology.


Implementation of the code is organised in Three Parts for classification task:

- First: Given an ontology OWL file we normalize it with Normalizer.groovy script using jcel jar. [Normalizer file could be found here](https://github.com/bio-ontology-research-group/el-embeddings)

	Command to Normalize: groovy -cp jcel.jar Normalizer.groovy -i <Input OWL ontology> -o <Output normalized-ontology> 
	
- Second: Using the normalized-ontology we identify the subclass relations and generate training, testing and validation set using
		   split of 70%-20%-10%.

- Third: Performing training using the normalized-ontology file while removing the 30%(validation and testing) subclass relation axioms from it.
		Using validation data for hyper-parameter tuning and testing to evaluate the fine-tuned models.
		
Associated model files:
- Experiments/models/EMEL_trans_m.py : This file denotes the EmEL model implementation with translation operation and variance.
- Experiments/models/EMEL_trans_bayes.py : This file denotes the EmEL model implementation with translation operation and bayesian inference.
- Experiments/models/EMEL_sparse.py : This file denotes the EmEL model implementation with relations as matrices.
- Experiments/models/EMEL_sparse_m.py : This file denotes the EmEL model implementation with relations as matrices and variance.
	
	
Executing the code:
- Before executing the code you need CUDA installed to use a GPU and list of python libraries as provided in requirements.txt.
- For execution of the code follow the directory structure as it is, further we demonstrate it using an example for GALEN dataset.
- Go to directory experiments/models/ folder and run python EMEL_trans_m.py --data GO (provide other arguments if needed)
- This will start the training and if you want to change the dimension size then you need to modify it in the code. 
- This will output corresponding embeddings for classes and relations in pkl files in the results directory.
- For evaluating the embeddings run python scripts Evaluating_HITS.py and provide the path of the pkl files.

## Data

The preprocessed data for Snomed, Galen and GO are present [here](https://doi.org/10.5281/zenodo.7023568)
	
Create Experiments/Data folder which would contain all the data.


