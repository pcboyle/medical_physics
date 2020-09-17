#pragma once
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <vector>

float min(float val_1, float val_2)
{
    return ((val_1) < (val_2)) ? (val_1) : (val_2);
}

float max(float val_1, float val_2)
{
    return ((val_1) > (val_2)) ? (val_1) : (val_2);
}

void add(float* v1, float* v2, float* v)
{
    v[0] = v1[0] + v2[0];
    v[1] = v1[1] + v2[1];
    v[2] = v1[2] + v2[2];
    v[3] = v1[3] + v2[3];
}

void subtract(float* v1, float* v2, float* v)
{
    v[0] = v1[0] - v2[0];
    v[1] = v1[1] - v2[1];
    v[2] = v1[2] - v2[2];
    v[3] = v1[3] - v2[3];
}

float dot(float* v1, float* v2)
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2] + v1[3] * v2[3];
}

float getMag(float* v)
{
    return sqrt(dot(v, v));
}

void getMagVect(float* v, float* result)
{
    float norm = sqrt(dot(v, v));
    result[0] = norm;
    result[1] = v[0];
    result[2] = v[1];
    result[3] = v[2];
    result[4] = v[3];
}

float getMag2(float* v, float* p)
{
    return sqrt((v[0] - p[0]) * (v[0] - p[0])
        + (v[1] - p[1]) * (v[1] - p[1])
        + (v[2] - p[2]) * (v[2] - p[2])
        + (v[3] - p[3]) * (v[3] - p[3]));
}

void getMag2Vect(float* v, float* p, float* result)
{
    float norm = sqrt((v[0] - p[0]) * (v[0] - p[0])
        + (v[1] - p[1]) * (v[1] - p[1])
        + (v[2] - p[2]) * (v[2] - p[2])
        + (v[3] - p[3]) * (v[3] - p[3]));

    result[0] = norm;
    result[1] = v[0] - p[0];
    result[2] = v[1] - p[1];
    result[3] = v[2] - p[2];
    result[4] = v[3] - p[3];
}

void getEdgeDist3(float* v, float* p1, float* p2, float* result)
{
    float nm[4], vec[4];
    subtract(p1, p2, nm);
    subtract(p1, v, vec);
    float m = dot(nm, nm);
    float d = dot(vec, nm);

    if (d > m)
    {
        getMag2Vect(v, p2, result);
        result[5] = -1;
    }
    else if (d < 0)
    {
        getMag2Vect(v, p1, result);
        result[5] = -1;
    }
    else
    {
        float dom = d / m;
        float f[4] = { p1[0] - dom * nm[0], p1[1] - dom * nm[1], p1[2] - dom * nm[2], p1[3] - dom * nm[3] };
        getMag2Vect(v, f, result);
        result[5] = -1;
    }
}

void inverse2(float m[2][2], float r[2][2])
{
    float det = m[0][0] * m[1][1] - m[0][1] * m[1][0];

    r[0][0] = m[1][1] / det;
    r[0][1] = -m[0][1] / det;
    r[1][0] = -m[1][0] / det;
    r[1][1] = m[0][0] / det;
}

void inverse3(float m[3][3], float r[3][3])
{
    float det = -m[0][2] * m[1][1] * m[2][0]
        + m[0][1] * m[1][2] * m[2][0]
        + m[0][2] * m[1][0] * m[2][1]
        - m[0][0] * m[1][2] * m[2][1]
        - m[0][1] * m[1][0] * m[2][2]
        + m[0][0] * m[1][1] * m[2][2];

    r[0][0] = (-m[1][2] * m[2][1] + m[1][1] * m[2][2]) / det;
    r[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
    r[0][2] = (-m[0][2] * m[1][1] + m[0][1] * m[1][2]) / det;

    r[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
    r[1][1] = (-m[0][2] * m[2][0] + m[0][0] * m[2][2]) / det;
    r[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;

    r[2][0] = (-m[1][1] * m[2][0] + m[1][0] * m[2][1]) / det;
    r[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
    r[2][2] = (-m[0][1] * m[1][0] + m[0][0] * m[1][1]) / det;
}

inline float threeWayMin(float d1, float d2, float d3)
{
    return d1 > d2 ? (d2 > d3 ? d3 : d2) : (d1 > d3 ? d3 : d2);
}

void assignVect5(float* v, float* result)
{
    result[0] = v[0];
    result[1] = v[1];
    result[2] = v[2];
    result[3] = v[3];
    result[4] = v[4];
}

void assignVect6(float* v, float* result)
{
    result[0] = v[0];
    result[1] = v[1];
    result[2] = v[2];
    result[3] = v[3];
    result[4] = v[4];
    result[5] = v[5];
}

void compareMag(float* v1, float* v2, float* result)
{
    if (v1[0] > v2[0])
    {
        assignVect5(v2, result);
    }
    else
    {
        assignVect5(v1, result);
    }
}

void compareMagM(float* v1, float* v2, float* result)
{
    if (v1[0] > v2[0])
    {
        assignVect6(v2, result);
    }
    else
    {
        assignVect6(v1, result);
    }
}

void compareThreeMag(float* v1, float* v2, float* v3, float* f)
{
    if ((v1[0] > v2[0]) && (v1[0] > v3[0]))
    {
        compareMagM(v2, v3, f);
    }

    else if ((v2[0] > v1[0]) && (v2[0] > v3[0]))
    {
        compareMagM(v1, v3, f);
    }

    else
    {
        compareMagM(v1, v2, f);
    }
}

void getTriangleDistOpt(float* v, float* p1, float* p2, float* p3, float* result, int i, int j, int k)
{
    float b[4], At[2][4];
    float compare1[6], compare2[6], compare3[6];
    subtract(v, p3, b);
    subtract(p1, p3, At[0]);
    subtract(p2, p3, At[1]);

    float Atb[2] = { dot(At[0], b), dot(At[1], b) };
    float AtA[2][2] = {
        {dot(At[0], At[0]), dot(At[0], At[1])},
        {dot(At[1], At[0]), dot(At[1], At[1])} };
    float iAtA[2][2];
    inverse2(AtA, iAtA);

    float wts[3];
    wts[0] = iAtA[0][0] * Atb[0] + iAtA[0][1] * Atb[1];
    wts[1] = iAtA[1][0] * Atb[0] + iAtA[1][1] * Atb[1];
    wts[2] = 1 - wts[0] - wts[1];

    if (wts[0] < 0.0f)
    {
        getEdgeDist3(v, p2, p3, compare1);
        if (wts[1] < 0.0f)  // two weights negative (can't be all negative)
        {
            getEdgeDist3(v, p1, p3, compare2);
            compareMagM(compare1, compare2, result);
        }
        else if (wts[2] < 0.0f) // two weights negative
        {
            getEdgeDist3(v, p1, p2, compare3);
            compareMagM(compare1, compare3, result);
        }
        else // one weight negative
        {
            assignVect6(compare1, result);
        }
    }
    else if (wts[1] < 0.0f)
    {
        getEdgeDist3(v, p1, p3, compare2);
        if (wts[2] < 0.0f) // two weights negative
        {
            getEdgeDist3(v, p1, p2, compare3);
            compareMagM(compare2, compare3, result);
        }
        else // one weight negative
        {
            assignVect6(compare2, result);
        }
    }
    else if (wts[2] < 0.0f) // one weight negative
    {
        getEdgeDist3(v, p1, p2, compare3);
        assignVect6(compare3, result);
    }
    else // all weights positive
    {
        // Return distance to this triangle
        float nv[4] = {
            wts[0] * At[0][0] + wts[1] * At[1][0] - b[0],
            wts[0] * At[0][1] + wts[1] * At[1][1] - b[1],
            wts[0] * At[0][2] + wts[1] * At[1][2] - b[2],
            wts[0] * At[0][3] + wts[1] * At[1][3] - b[3] };

        getMagVect(nv, result); // difference vector
        result[5] = 1;
    }
}

void getTetraDistOpt(float* v, float* p1, float* p2, float* p3, float* p4, float* result, float thresh, int i, int j, int k)
{
    /* test bounding box */

    float minf[4];
    for (int k = 0; k < 4; k++)
    {
        minf[k] = min(min(fabs(p1[k] - v[k]), fabs(p2[k] - v[k])), min(fabs(p3[k] - v[k]), fabs(p4[k] - v[k])));
    }
    if (getMag(minf) >= thresh)
    {
        result[0] = thresh;
    }

    float b[4], At[3][4];
    float compare1[6], compare2[6], compare3[6], compare4[6];
    subtract(v, p4, b);
    subtract(p1, p4, At[0]);
    subtract(p2, p4, At[1]);
    subtract(p3, p4, At[2]);

    float Atb[3] = { dot(At[0], b), dot(At[1], b), dot(At[2], b) };
    float AtA[3][3] = {
        {dot(At[0], At[0]), dot(At[0], At[1]), dot(At[0], At[2])},
        {dot(At[1], At[0]), dot(At[1], At[1]), dot(At[1], At[2])},
        {dot(At[2], At[0]), dot(At[2], At[1]), dot(At[2], At[2])} };
    float iAtA[3][3];
    inverse3(AtA, iAtA);

    float wts[4];
    wts[0] = iAtA[0][0] * Atb[0] + iAtA[0][1] * Atb[1] + iAtA[0][2] * Atb[2];
    wts[1] = iAtA[1][0] * Atb[0] + iAtA[1][1] * Atb[1] + iAtA[1][2] * Atb[2];
    wts[2] = iAtA[2][0] * Atb[0] + iAtA[2][1] * Atb[1] + iAtA[2][2] * Atb[2];
    wts[3] = 1 - wts[0] - wts[1] - wts[2];

    if (wts[0] < 0.0f)
    {
        getTriangleDistOpt(v, p2, p3, p4, compare1, i, j, k);
        if (wts[1] < 0.0f)
        {
            getTriangleDistOpt(v, p1, p3, p4, compare2, i, j, k);
            if (wts[2] < 0.0f) // three weights negative (can't all be negative)
            {
                getTriangleDistOpt(v, p1, p2, p4, compare3, i, j, k);
                compareThreeMag(compare1, compare2, compare3, result);
            }
            else if (wts[3] < 0.0f) // three weights negative
            {
                getTriangleDistOpt(v, p1, p2, p3, compare4, i, j, k);
                compareThreeMag(compare1, compare2, compare4, result);
            }
            else // two weights negative
            {
                compareMagM(compare1, compare2, result);
            }
        }
        else if (wts[2] < 0.0f)
        {
            getTriangleDistOpt(v, p1, p2, p4, compare3, i, j, k);
            if (wts[3] < 0.0f) // three weights negative
            {
                getTriangleDistOpt(v, p1, p2, p3, compare4, i, j, k);
                compareThreeMag(compare1, compare3, compare4, result);
            }
            else // two weights negative
            {
                compareMagM(compare1, compare3, result);
            }
        }
        else if (wts[3] < 0.0f) // two weights negative
        {
            getTriangleDistOpt(v, p1, p2, p3, compare4, i ,j ,k);
            compareMagM(compare1, compare4, result);
        }
        else // one weight negative
        {
            assignVect6(compare1, result);
        }
    }

    else if (wts[1] < 0.0f)
    {
        getTriangleDistOpt(v, p1, p3, p4, compare2, i, j, k);
        if (wts[2] < 0.0f)
        {
            getTriangleDistOpt(v, p1, p2, p4, compare3, i, j, k);
            if (wts[3] < 0.0f) // three weights negative
            {
                getTriangleDistOpt(v, p1, p2, p3, compare4, i, j, k);
                compareThreeMag(compare2, compare3, compare4, result);
            }
            else // two weights negative
            {
                compareMagM(compare2, compare3, result);
            }
        }
        else if (wts[3] < 0.0f) // two weights negative
        {
            getTriangleDistOpt(v, p1, p2, p3, compare4, i, j, k);
            compareMagM(compare2, compare4, result);
        }
        else // one weight negative
        {
            assignVect6(compare2, result);
        }
    }
    else if (wts[2] < 0.0f)
    {
        getTriangleDistOpt(v, p1, p2, p4, compare3, i, j, k);
        if (wts[3] < 0.0f) // two weights negative
        {
            getTriangleDistOpt(v, p1, p2, p3, compare4, i, j, k);
            compareMagM(compare3, compare4, result);
        }
        else // one weight negative
        {
            assignVect6(compare3, result);
        }
    }
    else if (wts[3] < 0.0f) // one weight negative
    {
        getTriangleDistOpt(v, p1, p2, p3, compare4, i, j, k);
        assignVect6(compare4, result);
    }
    else // all weights positive
    {
        // Return distance to this tetra
        float nv[4] = {
            wts[0] * At[0][0] + wts[1] * At[1][0] + wts[2] * At[2][0] - b[0],
            wts[0] * At[0][1] + wts[1] * At[1][1] + wts[2] * At[2][1] - b[1],
            wts[0] * At[0][2] + wts[1] * At[1][2] + wts[2] * At[2][2] - b[2],
            wts[0] * At[0][3] + wts[1] * At[1][3] + wts[2] * At[2][3] - b[3] };

        getMagVect(nv, result); // difference vector
        result[5] = 1;
    }
}

float calcTheta(float* v)
{
    float pi = 3.14159265f;
    float y = v[4]; // Value difference
    float x = sqrt(v[1] * v[1] + v[2] * v[2] + v[3] * v[3]); // Spatial difference

    return atan2(y, x) * 180 / pi;
}

//vect_1 is the reference, vect_2 is the comparison.
void findGammaPrint(std::vector <float>* vect_1, std::vector <float>* vect_2, float percent_diff, float dta, const char* output_file, int xsize, int ysize, int zsize)
{
    // Open output file
    std::ofstream output_data(output_file);

    std::ostringstream outbuffer;

    // Add size information and formatting to file. 
    for (int size_index = 0; size_index < 3; size_index++)
    {
        if (size_index == 0)
        {
            output_data << xsize << "\n";
        }
        if (size_index == 1)
        {
            output_data << ysize << "\n";
        }
        if (size_index == 2)
        {
            output_data << zsize << "\n";
        }
    }

    // Add variable names to top of each column. 
    const char* var_names[15] = { "Gamma", "Theta", "X Diff", "Y Diff" , "Z Diff", "DTA", "Rho Diff", 
                                  "X Calc", "Y Calc", "Z Calc", "Rho Calc", "X Ref", "Y Ref", "Z Ref", 
                                  "Rho Ref"};

    for (int vi = 0; vi < 15; vi++)
    {
        std::ostringstream value;
        value.width(10);
        value << var_names[vi];
        if (vi == 0)
        {
            outbuffer << value.str();
        }
        else
        {
            outbuffer << " " << value.str();
        }
    }
    outbuffer << "\n";

    // Get size. 
    size_t total_size = vect_1->size();
    
    // Set epsilon for considering lambda to be 0. 
    float epsilon = 0.001f;

    int margin = (int)ceil(dta); // 1
    int verts_per_side = 2 * margin + 1;
    float init_dist = (float)sqrt( (margin/dta)*(margin/dta) + (margin/dta)*(margin/dta) + (margin/dta)*(margin/dta) + (0.85f/percent_diff)*(0.85f/percent_diff));

    std::cout << "Margin set to " << margin << "\n";
    std::cout << "Max gamma set to " << init_dist << "\n";

    int tetra_points[5][4] = { {0,1,2,4},{2,4,6,7},{1,4,5,7},{1,2,3,7},{2,4,1,7} };
    float cell_tetra[8][4];

    std::cout << "Sorting cells." << "\n";
    // Sort the cells
    int n_cells = margin * margin * margin * 8;
    int n_verts_per_cell = 8;
    int n_vertices = n_cells * n_verts_per_cell;
    int* cell_vertex_x = new int[n_vertices] {0};
    int* cell_vertex_y = new int[n_vertices] {0};
    int* cell_vertex_z = new int[n_vertices] {0};
    int* x_shift = new int[n_vertices] {0};
    int* y_shift = new int[n_vertices] {0};
    int* z_shift = new int[n_vertices] {0};
    float* cell_dist = new float[n_cells] {0};
    int counter = 0;

    // If margin is greater than 1, sort cells, 
    // otherwise all cells have the same central physical distance. 
    for (int i = -margin; i < -margin + 2; i ++)
        for (int j = -margin; j < -margin + 2; j ++)
            for (int k = -margin; k < -margin + 2; k++)
            {
                cell_vertex_x[counter] = i;
                cell_vertex_y[counter] = j;
                cell_vertex_z[counter] = k;
                counter ++;
            }

    counter = 0;
    for (int i = -margin; i < margin; i++)
        for (int j = -margin; j < margin; j++)
            for (int k = -margin; k < margin; k++)
            {
                x_shift[counter] = i + margin;
                y_shift[counter] = j + margin;
                z_shift[counter] = k + margin;
                counter++;
            }

    for (int c_num = 0; c_num < n_cells; c_num++)
    {
        float avg_dist = 0;
        for (int v_num = 0; v_num < n_verts_per_cell; v_num++)
        {
            int cell_index = c_num * n_verts_per_cell + v_num;
            int x_loc = cell_vertex_x[v_num] + x_shift[c_num];
            int y_loc = cell_vertex_y[v_num] + y_shift[c_num];
            int z_loc = cell_vertex_z[v_num] + z_shift[c_num];
            cell_vertex_x[cell_index] = x_loc;
            cell_vertex_y[cell_index] = y_loc;
            cell_vertex_z[cell_index] = z_loc;

            avg_dist += (float)sqrt(x_loc * x_loc + y_loc * y_loc + z_loc * z_loc);
        }
        cell_dist[c_num] = avg_dist / n_verts_per_cell;
    }

    // Sort cells by distance. 
    for (int i = 0; i < n_cells - 1; i++)
        for (int j = n_cells - 2; j >= i; j--)
        {
            if (cell_dist[j] > cell_dist[j + 1])
            {
                float ftemp;
                int itemp;

                ftemp = cell_dist[j];
                cell_dist[j] = cell_dist[j + 1];
                cell_dist[j + 1] = ftemp;

                for (int v_num = 0; v_num < n_verts_per_cell; v_num++)
                {
                    int v_index_l = (j + 1) * n_verts_per_cell + v_num;
                    int v_index_h = j * n_verts_per_cell + v_num;

                    itemp = cell_vertex_x[v_index_l];
                    cell_vertex_x[v_index_h] = cell_vertex_x[v_index_l];
                    cell_vertex_x[v_index_l] = itemp;

                    itemp = cell_vertex_y[v_index_h];
                    cell_vertex_y[v_index_h] = cell_vertex_y[v_index_l];
                    cell_vertex_y[v_index_l] = itemp;

                    itemp = cell_vertex_z[v_index_h];
                    cell_vertex_z[v_index_h] = cell_vertex_z[v_index_l];
                    cell_vertex_z[v_index_l] = itemp;
                }
            }
        }

    std::cout << "Sorted cells." << "\n";

    // Loop through all voxels and find appropriate gamma. 
    for (int i = 0; i < xsize; i++)
    {
        if (i < xsize - 1)
        {
             std::cout << "Slice " << i + 1 << " of " << xsize << "\r";
        }
        else
        {
            std::cout << "Slice " << i + 1 << " of " << xsize << "\n";
        }
        for (int j = 0; j < ysize; j++)
        {
            for (int k = 0; k < zsize; k++)
            {
                int data_index = i * ysize * zsize + j * zsize + k;

                int border = 0;
                if (i < margin || j < margin || k < margin || i >= xsize - margin || j >= ysize - margin || k >= zsize - margin)
                {
                    border = 1;
                }

                // Set dist to margin as initial value. 
                float dist = init_dist;
                float gamma_vector[6] = { init_dist, margin/dta, margin/dta, margin/dta, 0.85f/percent_diff, 1};
                int best_cell = 0;

                float ref_value = vect_1->at(data_index);
                float comp_value = vect_2->at(data_index);
                float ref_vector[4] = { i / dta, j / dta, k / dta, ref_value / percent_diff };

                float zero_vect[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                if (abs(ref_value - comp_value)/percent_diff <= epsilon)
                {
                    dist = 0.0f;
                    assignVect6(zero_vect, gamma_vector);
                }

                else
                {
                    for (int l = 0; l < n_cells; l++)
                    {
                        int lowest_cell_vertex = l * n_verts_per_cell;

                        int x = i + cell_vertex_x[lowest_cell_vertex];
                        int y = j + cell_vertex_y[lowest_cell_vertex];
                        int z = k + cell_vertex_z[lowest_cell_vertex];

                        if (border && (x < 0 || x > xsize - 2 || y < 0 || y > ysize - 2 || z < 0 || z > zsize - 2))
                        {
                            continue;
                        }

                        if (cell_dist[l] >= dist)
                        {
                            break;
                        }

                        // Initialize variables. 
                        float m_vect[6];
                        float current_separation[5];
                        float mean_center[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                        float min_vect[5] = { 1000000.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                        float msum = 0.0f;

                        // Get 8 vertices for each cell. 
                        for (int index = 0; index < 8; index++)
                        {
                            int vert_index = l * n_verts_per_cell + index;
                            int ix = i + cell_vertex_x[vert_index];
                            int iy = j + cell_vertex_y[vert_index];
                            int iz = k + cell_vertex_z[vert_index];

                            float cs_min[5];
                            int data_index_2 = ix * ysize * zsize + iy * zsize + iz;

                            cell_tetra[index][0] = ix / dta;
                            cell_tetra[index][1] = iy / dta;
                            cell_tetra[index][2] = iz / dta;
                            cell_tetra[index][3] = vect_2->at(data_index_2) / percent_diff;

                            getMag2Vect(ref_vector, cell_tetra[index], current_separation);

                            msum += current_separation[0];

                            compareMag(min_vect, current_separation, cs_min);
                            assignVect5(cs_min, min_vect);

                            mean_center[0] += cell_tetra[index][0];
                            mean_center[1] += cell_tetra[index][1];
                            mean_center[2] += cell_tetra[index][2];
                            mean_center[3] += cell_tetra[index][3];
                        }

                        mean_center[0] /= 8.0f;
                        mean_center[1] /= 8.0f;
                        mean_center[2] /= 8.0f;
                        mean_center[3] /= 8.0f;
                        float mean_center_distance = getMag2(ref_vector, mean_center);

                        // Determines best point via tetrahedral simplexes. 
                        for (int t = 0; t < 5; t++)
                        {
                            getTetraDistOpt(ref_vector, cell_tetra[tetra_points[t][0]], cell_tetra[tetra_points[t][1]], cell_tetra[tetra_points[t][2]], cell_tetra[tetra_points[t][3]], m_vect, init_dist, i, j, k);
                            if (m_vect[0] < dist)
                            {
                                dist = m_vect[0];
                                assignVect6(m_vect, gamma_vector);
                            }
                        }
                    }
                }

                // Build array to hold x, y, z distances, and gammas. 
                float final_out[15];
                final_out[0] = gamma_vector[0]; // gamma

                float theta = 0.0f;
                if (gamma_vector[0] > 0.0000009f)
                {
                    theta = calcTheta(gamma_vector);
                }
                final_out[1] = theta;

                final_out[2] = gamma_vector[1]*dta; // dta weighted x difference
                final_out[3] = gamma_vector[2]*dta; // dta weighted y difference
                final_out[4] = gamma_vector[3]*dta; // dta weighted z difference
                final_out[5] = sqrt(gamma_vector[1] * gamma_vector[1] + gamma_vector[2] * gamma_vector[2] + gamma_vector[3] * gamma_vector[3])*dta; // dta weights spatial magnitude
                final_out[6] = gamma_vector[4]*percent_diff; // percent_diff weighted value difference

                final_out[7] = (ref_vector[0] + gamma_vector[1]*gamma_vector[5]) * dta; // point x
                final_out[8] = (ref_vector[1] + gamma_vector[2]*gamma_vector[5]) * dta; // point y
                final_out[9] = (ref_vector[2] + gamma_vector[3]*gamma_vector[5]) * dta; // point z
                final_out[10] = (ref_vector[3] + gamma_vector[4]*gamma_vector[5]) * percent_diff; // point value

                final_out[11] = (ref_vector[0]) * dta; // ref x
                final_out[12] = (ref_vector[1]) * dta; // ref y
                final_out[13] = (ref_vector[2]) * dta; // ref z
                final_out[14] = (ref_vector[3]) * percent_diff; // ref value

                // Print value to outbuffer to avoid holding large multidimensional array. 
                for (int vi = 0; vi < 15; vi++)
                {
                    std::ostringstream value;
                    value << std::fixed;
                    value.precision(6);
                    value.width(10);
                    value << final_out[vi];
                    if (vi == 0)
                    {
                        outbuffer << value.str();
                    }
                    else
                    {
                        outbuffer << " " << value.str();
                    }
                }
                outbuffer << "\n";
            }
        }
    }

    output_data << outbuffer.str();

    // Delete created variables. 
    delete[] cell_vertex_x;
    delete[] cell_vertex_y;
    delete[] cell_vertex_z;
    delete[] x_shift;
    delete[] y_shift;
    delete[] z_shift;
    delete[] cell_dist;

    output_data.close();
}