#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

using namespace std;

void generateInputFile(const string& fileName, int size) {
    ofstream outputFile(fileName);

    if (!outputFile.is_open()) {
        cerr << "Error opening the file: " << fileName << endl;
        exit(1);
    }

    srand(static_cast<unsigned>(time(nullptr)));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            outputFile << rand() % 10 + 1 << " ";
        }
        outputFile << endl;
    }

    outputFile.close();
}

int main() {
    string fileName = "input2048.txt";
    int size = 2048; 

    generateInputFile(fileName, size);

    cout << "Generated input file: " << fileName << " with " << size << " vertices." << endl;

    return 0;
}

