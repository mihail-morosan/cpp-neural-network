#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int InputCount, int HiddenLayerNeuronCount, int OutputCount, bool SigmoidActivation)
{
	//Well. My experiment with Linear Activation function sort of failed.
	IsSigmoidActivation = SigmoidActivation;

	//Initialise as a single-hidden layer NN
	vector<Neuron> InputLayer(InputCount+1);
	vector<Neuron> HiddenLayer(HiddenLayerNeuronCount+1);
	vector<Neuron> OutputLayer(OutputCount);

	InputLayer[0].SetOutput(1); //Input Bias
	HiddenLayer[0].SetOutput(1); //Hidden Layer Bias

	for(int i=1; i<=InputCount; i++)
	{
		//Initialise
	}

	for(int i=1; i<=HiddenLayerNeuronCount; i++)
	{
		HiddenLayer[i].SetRandomWeights(InputCount+1);
		HiddenLayer[i].SetSigmoidActivation(SigmoidActivation);
	}

	for(int i=0; i<OutputCount; i++)
	{
		OutputLayer[i].SetRandomWeights(HiddenLayerNeuronCount+1);
		OutputLayer[i].SetSigmoidActivation(SigmoidActivation);

	}

	Network.push_back(InputLayer);
	Network.push_back(HiddenLayer);
	Network.push_back(OutputLayer);

	Error.assign(OutputCount,0.0f);
	ValidationRuns = 0;
	ValidationTotalError.assign(OutputCount,0.0f);

}

bool NeuralNetwork::RunDataSet(vector<float> &Inputs, vector<float> &DesiredOutputs)
{
	if(Network.size() < 3)
	{
		return false;
	}
	if(Network[0].size() != Inputs.size()+1)
	{
		return false;
	}
	if(Network[2].size() != DesiredOutputs.size())
	{
		return false;
	}

	for(unsigned int i=0; i<Inputs.size(); i++)
	{
		Network[0][i+1].SetOutput(Inputs[i]);
	}

	//Feed-forward processing
	for(int i=1; i<3; i++)
	{
		for(unsigned int y=0; y<Network[i].size(); y++)
		{
			if(i!=1 || y!=0)
			{
				Network[i][y].CalculateOutput(Network[i-1]);
			}
		}
	}

	//Calculate Error
	for(unsigned int i=0; i<DesiredOutputs.size(); i++)
	{
		Error[i] = DesiredOutputs[i] - Network[2][i].GetOutput();
	}

	return true;
}

void NeuralNetwork::ValidateRun(vector<float> &Inputs, vector<float> &DesiredOutputs)
{
	if(!RunDataSet(Inputs, DesiredOutputs))
		return;

	ValidationRuns++;
	for(unsigned int i=0; i<Error.size(); i++)
	{
		//ValidationTotalError[i]+=(Error[i] * Error[i]) * 1.0f/2.0f;
		ValidationTotalError[i]+=(Error[i] * Error[i]);
	}
}

float NeuralNetwork::GetValidationAverageError()
{
	if(ValidationRuns > 0)
	{
		float result = 0;
		for(unsigned int i=0; i<Error.size(); i++)
		{
			result += ValidationTotalError[i] / ValidationRuns;
		}
		result = result / Error.size();
		return result;
	} else {
		return -1;
	}
}

void NeuralNetwork::ClearHistoricalErrorData()
{
	HistoricalError.clear();
	for(unsigned int i=0; i<Error.size(); i++)
	{
		ValidationTotalError[i] = 0;
	}
	ValidationRuns = 0;
}

float NeuralNetwork::GetHistoricalDataAverageError()
{
	float result = 0;
	if(HistoricalError.size()==0)
	{
		return 0;
	}
	for(unsigned int i=0; i<HistoricalError.size(); i++)
	{
		//result += HistoricalError[i] * HistoricalError[i] * 1.0f/2.0f;
		result += HistoricalError[i];
	}
	result = result / HistoricalError.size();
	return result;
}

void NeuralNetwork::TrainNetwork(vector<float> &Inputs, vector<float> &DesiredOutputs)
{
	if(!RunDataSet(Inputs, DesiredOutputs))
		return;

	//Update Historical data
	float result = 0.0f;
	for(unsigned int i=0; i<Error.size(); i++)
	{
		//result += abs(Error[i]);
		result += Error[i]*Error[i];
	}
	HistoricalError.push_back(result / Error.size());

	//Calculate Local Gradients
	//Output layer
	for(unsigned int i=0; i<Network[2].size(); i++)
	{
		Network[2][i].CalculateLocalGradientO(Error[i]);
	}
	//Hidden layer
	for(unsigned int i=1; i<Network[1].size(); i++)
	{
		Network[1][i].CalculateLocalGradientH(Network[2], i);
	}

	//Update all weights
	//Output layer
	for(unsigned int i=0; i<Network[2].size(); i++)
	{
		Network[2][i].UpdateWeightsFromGradient(Network[1]);
	}
	//Hidden layer
	for(unsigned int i=1; i<Network[1].size(); i++)
	{
		Network[1][i].UpdateWeightsFromGradient(Network[0]);
	}

}

vector<float> NeuralNetwork::GetOutput()
{
	vector<float> r;
	for(unsigned int i=0; i<Network[2].size(); i++)
	{
		r.push_back(Network[2][i].GetOutput());
	}
	return r;
}

void NeuralNetwork::SaveWeightsToFile(char* filename)
{
	ofstream f(filename);

	for(unsigned int i=1; i<Network.size(); i++)
	{
		for(unsigned int y=0; y<Network[i].size(); y++)
		{
			if(i!=1 || y!=0)
			{
				vector<float> w = Network[i][y].GetWeights();
				for(unsigned int z=0; z<w.size(); z++)
				{
					f<<w[z]<<",";
				}
			}
		}
	}

	f.close();
}

void NeuralNetwork::LoadWeightsFromFile(char* filename)
{
	ifstream f(filename);

	char b;
	for(unsigned int i=1; i<Network.size(); i++)
	{
		for(unsigned int y=0; y<Network[i].size(); y++)
		{
			if(i!=1 || y!=0)
			{
				float w;
				vector<float> wr;
				for(unsigned int z=0; z<Network[i-1].size(); z++)
				{
					f>>w>>b;
					wr.push_back(w);
				}
				Network[i][y].SetWeights(wr);
			}
		}
	}

	f.close();
}