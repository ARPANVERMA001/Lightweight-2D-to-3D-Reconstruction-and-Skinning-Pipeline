#ifndef FILE_IO_UTILS_H
#define FILE_IO_UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <array>
#include <stdexcept>
#include <Eigen/Dense>

/**
 * Cleans a line starting with 'n', removing specific characters based on the Python logic.
 * @param input The input string.
 * @return Cleaned string.
 */
std::string clean_nan_line(const std::string &input);

/**
 * Loads a DMAT file and returns it as an Eigen::MatrixXd.
 * @param path The path to the DMAT file.
 * @return The loaded matrix.
 */
Eigen::MatrixXd load_DMAT(const std::string &path);

/**
 * Writes an Eigen::MatrixXd to a DMAT file.
 * @param path The output file path.
 * @param M The matrix to write.
 */
void write_DMAT(const std::string &path, const Eigen::MatrixXd &M);

/**
 * Loads a Tmat file into an Eigen::MatrixXd.
 * @param path The path to the Tmat file.
 * @return The loaded matrix.
 */
Eigen::MatrixXd load_Tmat(const std::string &path);

/**
 * Loads poses from a file and returns them as a nested vector of 3D arrays.
 * @param path The path to the poses file.
 * @return The loaded poses.
 */
std::vector<std::vector<std::array<double, 3>>> load_poses(const std::string &path);

/**
 * Loads results and weights from a file.
 * @param path The path to the results file.
 * @return A pair containing a nested vector of arrays for results and an Eigen::MatrixXd for weights.
 */
std::pair<std::vector<std::vector<std::array<double, 12>>>, Eigen::MatrixXd> load_result(const std::string &path);

/**
 * Writes results and weights to output files.
 * @param path_res The path to the results file.
 * @param path_w The path to the weights file.
 * @param res The results matrix.
 * @param weights The weights matrix.
 * @param iter_num The iteration number.
 * @param time The time taken.
 * @param col_major Whether to write in column-major order.
 */
void write_result(const std::string &path_res, const std::string &path_w,
                   const Eigen::MatrixXd &res, const Eigen::MatrixXd &weights,
                   int iter_num, double time, bool col_major = false);

/**
 * Writes vertex and face data to an OBJ file.
 * @param path The path to the OBJ file.
 * @param vs The vertex data.
 * @param fs The face data.
 */
void write_OBJ(const std::string &path,
               const std::vector<std::array<double, 3>> &vs,
               const std::vector<std::array<int, 3>> &fs);

#endif // FILE_IO_UTILS_H