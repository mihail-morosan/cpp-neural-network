#include "Neuron.h"
#include "Helpers.h"

Neuron::Neuron()
{
	//Weights.assign(PrevLayerNeuronCount, 0.0f);
	
}

void Neuron::SetSigmoidActivation(bool IsIt)
{
	IsSigmoidActivation = IsIt;
}

void Neuron::SetWeights(vector<float>& newWeights)
{
	Weights = newWeights;
}

void Neuron::SetRandomWeights(int Number)
{
	vector<float> w(Number);
	for(int i=0; i<Number; i++)
	{
		w[i] = Helpers::RandomValue01();
		//w[i] = Helpers::RandomValueMinus11();
	}
	SetWeights(w);
	PreviousWeightDelta.assign(Number,0);
}

void Neuron::CalculateOutput(vector<Neuron> &previousLayer)
{
	//Calculate here
	float result = 0.0f;

	for(unsigned int i=0; i<previousLayer.size(); i++)
	{
		result += previousLayer[i].GetOutput() * Weights[i];
	}

	if(IsSigmoidActivation)
	{
		if (result < -45.0) 
			result = 0;
		else if (result > 45.0) 
			result = 1;
		else
			result = 1.0f / ( 1.0f + exp(-result) );
	}

	SetOutput(result);
}

float Neuron::GetOutput()
{
	return Output;
}

void Neuron::CalculateLocalGradientO(float error)
{
	if(IsSigmoidActivation)
	{
		LocalGradient = Lambda * Output * (1 - Output) * error;
	} else {
		LocalGradient = error;
	}
}

void Neuron::CalculateLocalGradientH(vector<Neuron> &NextLayer, int index)
{
	if(IsSigmoidActivation)
	{
		LocalGradient = Lambda * Output * (1 - Output);
	} else {
		LocalGradient = 1;

	}

	//TODO: Problem might be here
	float r = 0;
	for(unsigned int i=0; i<NextLayer.size(); i++)
	{
		r += NextLayer[i].GetLocalGradient() * NextLayer[i].Weights[index];
	}

	LocalGradient *= r;
}

float Neuron::GetLocalGradient()
{
	return LocalGradient;
}

void Neuron::SetOutput(float output)
{
	Output = output;
}

//http://zerkpage.tripod.com/backprop.txt
//http://msdn.microsoft.com/en-us/magazine/jj658979.aspx

void Neuron::UpdateWeightsFromGradient(vector<Neuron> &previousLayer)
{
	for(unsigned int i=0; i<Weights.size(); i++)
	{
		float delta = Eta * LocalGradient * previousLayer[i].GetOutput();
		if(PreviousWeightDelta[i]!=0)
		{
			Weights[i] = Weights[i] + delta + Alpha * PreviousWeightDelta[i];
		} else {
			Weights[i] = Weights[i] + delta;
		}
		PreviousWeightDelta[i] = delta;
	}
}
