#include <iostream>
#include <cmath>
#include <vector>

// Function to calculate using the numerical Jacobi method
/**
 * Solves a system of linear equations using the Jacobi iterative method.
 *
 * @param A The coefficient matrix of the system of linear equations.
 * @param b The right-hand side vector of the system of linear equations.
 * @param tolerance The desired tolerance for the solution.
 * @param maxIterations The maximum number of iterations to perform.
 * @return The solution vector.
 */
std::vector<double> jacobiMethod(const std::vector<std::vector<double>> &A, const std::vector<double> &b, double tolerance, int maxIterations)
{
    int n = A.size();
    std::vector<double> x(n, 0.0); // Initial guess
    std::vector<double> xNew(n);

    int iteration = 0;
    double error = tolerance + 1.0;

    while (error > tolerance && iteration < maxIterations)
    {
        error = 0.0;
        for (int i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    sum += A[i][j] * x[j];
                }
            }
            xNew[i] = (b[i] - sum) / A[i][i];
            error += std::abs(xNew[i] - x[i]);
        }
        x = xNew;
        iteration++;
    }

    return x;
}

// Function to calculate the equation mathematically
/**
 * Solves a system of linear equations Ax = b using Gaussian elimination.
 *
 * @param A The coefficient matrix.
 * @param _b The right-hand side vector.
 * @return The solution vector x.
 */
std::vector<double> calculateEquation(std::vector<std::vector<double>> &A, const std::vector<double> &_b)
{
    std::vector<double> b = _b;
    int n = A.size();
    std::vector<double> x(n);

    // Perform Gaussian elimination
    for (int i = 0; i < n; i++)
    {
        // Find pivot element
        int pivot = i;
        for (int j = i + 1; j < n; j++)
        {
            if (std::abs(A[j][i]) > std::abs(A[pivot][i]))
            {
                pivot = j;
            }
        }

        // Swap rows if necessary
        if (pivot != i)
        {
            std::swap(A[i], A[pivot]);
            std::swap(b[i], b[pivot]);
        }

        // Eliminate variables
        for (int j = i + 1; j < n; j++)
        {
            double factor = A[j][i] / A[i][i];
            for (int k = i + 1; k < n; k++)
            {
                A[j][k] -= factor * A[i][k];
            }
            b[j] -= factor * b[i];
        }
    }

    // Back-substitution
    for (int i = n - 1; i >= 0; i--)
    {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++)
        {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
    }

    return x;
}

/**
 * The main entry point of the program.
 * This function sets up the coefficients matrix A and the constant vector b,
 * then calculates the solution using both the Jacobi method and the mathematical equation.
 * The results are printed to the console.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of C-style strings containing the command-line arguments.
 * @return 0 on successful completion.
 */
int main(int argc, char *argv[])
{
    // Define the coefficients matrix A and the constant vector b
    std::vector<std::vector<double>> A = {{10, 2, 1}, {1, 5, 1}, {2, 3, 10}};
    std::vector<double> b = {7, -8, 6};

    // Calculate the solution using the Jacobi method
    std::vector<double> xJacobi = jacobiMethod(A, b, 1e-6, 1000);
    std::cout << "Solution using Jacobi method: ";
    for (double x : xJacobi)
        std::cout << x << " ";
    std::cout << std::endl;

    // Calculate the solution using the mathematical equation
    std::vector<double> xEquation = calculateEquation(A, b);
    std::cout << "Solution using mathematical equation: ";
    for (double x : xEquation)
        std::cout << x << " ";
    std::cout << std::endl;

    return 0;
}
