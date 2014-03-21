#include <vector>
#include <fstream>
#include "Neuron.h"

using namespace std;

class NeuralNetwork
{
	vector<vector<Neuron>> Network;
	vector<float> Error;

	vector<float> HistoricalError;

	vector<float> ValidationTotalError;
	int ValidationRuns;
	bool IsSigmoidActivation;
public:
	NeuralNetwork(int InputCount, int HiddenLayerNeuronCount, int OutputCount, bool SigmoidActivation);
	bool RunDataSet(vector<float> &Inputs, vector<float> &DesiredOutputs);
	void ValidateRun(vector<float> &Inputs, vector<float> &DesiredOutputs);
	void TrainNetwork(vector<float> &Inputs, vector<float> &DesiredOutputs);

	float GetValidationAverageError();
	float GetHistoricalDataAverageError();

	void ClearHistoricalErrorData();

	vector<float> GetOutput();

	void SaveWeightsToFile(char* filename);
	void LoadWeightsFromFile(char* filename);
};