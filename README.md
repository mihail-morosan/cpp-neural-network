cpp-neural-network
==================

Simple single-hidden-layer perceptron implemented in C++.

Example usage: 

NeuralNetwork* nn = new NeuralNetwork(INPUTCOUNT,HIDDENNEURONS,OUTPUTCOUNT,true);
 
vector<float> Input;
vector<float> DOutput;
 
//Store inputs in Input and desired outputs (for training) in DOutput
//You then have access to the following functions
nn->TrainNetwork(Input, DOutput); //Uses the input and desired output to train the network
nn->ValidateRun(Input, DOutput);  //Used for validation. Most probable use would be for early-stopping of any training
 
float EpochTrainingError = nn->GetHistoricalDataAverageError(); //Gets the root-mean-error for all the training sets passed to the network
float EpochValidationError = nn->GetValidationAverageError();   //Gets the root-mean-error for all the validation sets passed to the network
nn->ClearHistoricalErrorData(); //Clears the data retrieved above
 
nn->RunDataSet(Input, DOutput); //Uses the input and runs it through the network. Does not use DOutput at all
nn->GetOutput(); //Returns the output of the neural network after being passed an input set. To be used when you're confident in the neural network's prediction accuracy