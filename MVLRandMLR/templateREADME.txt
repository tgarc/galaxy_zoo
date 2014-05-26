README

The regression template can be used for any model. Just make sure KNOB's
and CHANGE's sections of the model are modified appropriately. To search
for the different KNOB's and CHANGE's, make sure the Match Case box is 
checked.

KNOBS: Select what to perform on each model
KNOB1 - (true/false) set whether to build models on the leaves 
	(37 models - 1 prediction per leaf)
KNOB2 - (true/false) set whether to model on the nodes 
	(11 models - # of leaves predictions per node)
KNOB3 - (true/false)set whether to model on the tree 
	(1 model - 37 predictions)
KNOB4 - (true/false) set whether to sample if building on leaves
KNOB5 - (num) select the number of samples if sampling
KNOB6 - (true/false) select whether to apply bagging for sampling
KNOB7 - (num) select how many times to build sample models (to lower bias)
KNOB8 - (true/false) select whether to compute scaled error or not

CHANGE: Related to type of model
CHANGE1 - set working directory
CHANGE2 - libraries required for model
CHANGE3 - formula inserted in model (only applicable to certain models)
CHANGE4 - model
CHANGE5 - train prediction (change the inserts)
CHANGE6 - test prediction (change the inserts)
CHANGE7 - write the train prediction
CHANGE8 - write the test prediction
CHANGE9 - write the error