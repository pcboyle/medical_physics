#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <vector>

#include "Header.h"

int main(int arg_count, char* arg_values[])
{
    // Clear buffer. 
    std::cout << std::flush;
    clock_t start, finish;
    start = clock();

    // Setting default strings to prevent nullptr errors. 
    const char* input_file_1 = "in_1.txt";
    const char* input_file_2 = "in_2.txt";
    const char* output_file = "out.txt";

    if (arg_values[1] != nullptr)
    {
        input_file_1 = arg_values[1];
        input_file_2 = arg_values[2];
        output_file = arg_values[3];
    }

    std::cout << "Reading " << input_file_1 << " and " << input_file_2 << "." << "\n";
    std::cout << "Writing to " << output_file << "." << "\n";

    // Initialize these to 3.0% and 3.0 pixels.
    float rho_percentage = 3.0f;
    float dta = 3.0f;

    if (arg_count < 5)
    {
        std::cout << "Setting rho percentage and DTA to 3.0% and 3.0 voxels." << "\n";
        rho_percentage /= 100.0f;
    }

    else
    {
        rho_percentage = (float)std::stof(arg_values[4]);
        dta = (float)std::stof(arg_values[5]);

        std::cout << "Setting rho percentage and DTA to " << rho_percentage << "%" << " and " << dta << " voxels." << "\n";

        // Make this a percentage. 
        rho_percentage /= 100.0f;
    }

    std::ifstream data_1(input_file_1);
    std::ifstream data_2(input_file_2);

    // Variables for the line for each data file. 
    std::string data_1_line;
    std::string data_2_line;

    // Initialize x, y, z size and set line_num to -1.
    int xsize_1 = 0, ysize_1 = 0, zsize_1 = 0;
    int xsize_2 = 0, ysize_2 = 0, zsize_2 = 0;

    int line_num = 0;

    // Read in the x, y, and z sizes of each file. 

    while (std::getline(data_1, data_1_line) && std::getline(data_2, data_2_line) && line_num++ <= 2)
    {
        if (line_num == 1)
        {
            xsize_1 = std::atoi(data_1_line.c_str());
            xsize_2 = std::atoi(data_2_line.c_str());
        }
        if (line_num == 2)
        {
            ysize_1 = std::atoi(data_1_line.c_str());
            ysize_2 = std::atoi(data_2_line.c_str());
        }
        if (line_num == 3)
        {
            zsize_1 = std::atoi(data_1_line.c_str());
            zsize_2 = std::atoi(data_2_line.c_str());
            break;
        }
    }
    // Now data lines are at the start of the data. 

    // Check input file sizes. 
    if (xsize_1 != xsize_2 || ysize_1 != ysize_2 || zsize_1 != zsize_2)
    {
        std::cout << "File sizes do not match!" << "\n";
        std::cout << "x sizes: " << xsize_1 << ", " << xsize_2 << "\n";
        std::cout << "y sizes: " << ysize_1 << ", " << ysize_2 << "\n";
        std::cout << "z sizes: " << zsize_1 << ", " << zsize_2 << "\n";
        return 0;
    }

    std::cout << "xsize, ysize, and zsize are: " << xsize_1 << ", " << ysize_1 << ", and " << zsize_1 << "\n";

    // Initialize vectors to hold values from input files. 
    int total_length = xsize_1 * ysize_1 * zsize_1;

    std::vector <float> value_vector_1(total_length);
    std::vector <float> value_vector_2(total_length);

    int vector_index = 0;
    while (std::getline(data_1, data_1_line) && std::getline(data_2, data_2_line))
    {
        float line_value_1 = (float)std::stod(data_1_line);
        value_vector_1.at(vector_index) = line_value_1;

        float line_value_2 = (float)std::stod(data_2_line);
        value_vector_2.at(vector_index) = line_value_2;
        vector_index++;
    }

    // Close data files now that we are done with them. 
    data_1.close();
    data_2.close();

    finish = clock();

    std::cout << "Files read! Time elapsed: " << (float)(finish - start) / (float)CLOCKS_PER_SEC << " seconds." << "\n";

    // Now need to determine gammas, thetas, and write to output file!
    findGammaPrint(&value_vector_1, &value_vector_2, rho_percentage, dta, output_file, xsize_1, ysize_1, zsize_1);

    finish = clock();

    std::cout << "Operation complete! Time elapsed: " << (float)(finish - start) / (float)CLOCKS_PER_SEC << " seconds." << "\n";

    return 0;
}