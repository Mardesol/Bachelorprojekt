#include "..\..\Header Files\matrixCompatability.cuh"

bool isCompatibleForAddition(int M1Rows, int M1Cols, int M2Rows, int M2Cols) {
	if (M1Rows == M2Rows && M1Cols == M2Cols) {
		return true;
	}
	else return false;
}
bool isCompatibleForMultiplication(int M1Cols, int M2Rows) {
	if (M1Cols == M2Rows) {
		return true;
	}
	else return false;
}