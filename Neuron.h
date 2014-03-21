#include <vector>
#include <math.h>

using namespace std;

const float Lambda = 0.8f; //In local gradient
const float Eta = 0.9f; //In weight updating
const float Alpha = 0.04f; //Momentum

class Neuron
{
	vector<float> Weights;
	float Output;
	float LocalGradient;
	vector<float> PreviousWeightDelta;
	bool IsSigmoidActivation;

public:
	
	Neuron();
	void SetSigmoidActivation(bool IsIt);
	void SetWeights(vector<float>& newWeights);
	void SetRandomWeights(int Number);
	void UpdateWeightsFromGradient(vector<Neuron> &previousLayer);
	void CalculateOutput(vector<Neuron> &previousLayer);
	float GetLocalGradient();
	void CalculateLocalGradientO(float error);
	void CalculateLocalGradientH(vector<Neuron> &NextLayer, int index);
	void SetOutput(float output);
	float GetOutput();
	vector<float> GetWeights() { return Weights; }
};
