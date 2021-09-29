# EmELpp
EmEL++ is a geometric approach to generate embeddings for the description logic EL++
The implementation is done using Python and Tensorflow Library.

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
tensorboard 1.15.0     
tensorflow-gpu 1.15.0 
torch 1.5.0

The code is organized as follows:
- experiments: This contains separate folder for each ontology the experiment is carried out upon.
- The implementation of evaluation metrics - Evaluating_HITS.py and Evaluating_accuracy.py


Since each of the ontologies have different characteristics based on normal forms 
thus inputs to the model defined using Keras change. Thus, we organise the implementation per ontology.

For Example:
-experiments/GALEN/GALEN_EmEL.py : This file represents the implementation of EmEL++ model 
for GALEN ontology.
-experiments/GALEN/GALEN_EL.py : This file represents the implementation of ElEm model 
for GALEN ontology.

-experiments/GALEN/data/ : This folder consists of 4 processed files namely, normalized form of the ontology file
to be used for training, and training,validation & testing set obtained from subclass relations in ontology.

Similary, the corresponding files are organised for NCI,GO,FMA,ANATOMY and SNOMED ontology.

Implementation of the code is organised in Three Parts for classification task:

- First: Given an ontology OWL file we normalize it with Normalizer.groovy script using jcel jar.
	Command to Normalize: groovy -cp jcel.jar Normalizer.groovy -i <Input OWL ontology> -o <Output normalized-ontology>

	
- Second: Using the normalized-ontology we identify the subclass relations and generate training, testing and validation set using
		   split of 70%-20%-10%.

- Third: Performing training using the normalized-ontology file while removing the 30%(validation and testing) subclass relation axioms from it.
		Using validation data for hyper-parameter tuning and testing to evaluate the fine-tuned models.
		
Associated Files:
- Train_valid_test_set_generation.py : This file covers the second part mentioned above.
- <OntologyName>_EmEL.py : This file denotes the EmEL++ model implementation. for example: GALEN_EmEL.py

Executing the code:
- Before executing the code you need CUDA installed to use a GPU and list of python libraries as provided in requirements.txt.
- For execution of the code follow the directory structure as it is, further we demonstrate it using an example for GALEN dataset.
- Go to directory experiments/GALEN
- Run GALEN_EmEL.py using the command: python GALEN_EmEL.py, This will start the training and if you want to change the hyper-parameter
values or path of data one needs to specify it in the file. 
- This will output corresponding embeddings for classes and relations in pkl files.
- For evaluating the embeddings run python scripts Evaluating_accuracy.py and Evaluating_accuracy.py provide the path of the pkl files.

Similary, you can do the process for other ontologies present.


## Data

The preprocessed data for Snomed, Galen and GO are present [here](https://drive.google.com/file/d/1HHXogBAaQ5f0lJrAI_fVkuAmSTuh2IQW/view?usp=sharing)


