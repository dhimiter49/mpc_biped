#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>


// Function to multiply matrix and vector
void matVecMult3(const double mat[3][3], const double vec[3], double result[3]) {
    for (int i = 0; i < 3; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < 3; ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }
}


#include <qpOASES.hpp>


/** Example for qpOASES main function using the QProblem class. */
int main( )
{
    USING_NAMESPACE_QPOASES

    const double max_constraint = 0.17;
    const double min_constraint = -max_constraint;
    const double max_min_constraint = -0.03;
    const double min_max_constraint = -max_min_constraint;
    const double start_time = 2.5; // in s
    const double end_time = 7.5; // in s
    const double grav = 9.81; // gravity
    const double h_com = 0.8; // height of CoM
    const double dt = 0.005; // in s
    const int N = 300; // lookahead
    const double T = 9; // in s
    const int period = 1 / dt; // in s
    const int short_period = 0.8 / dt; // in s
    const int diff_period = (period - short_period) / 2;

    // Iteration dynamics
    double dyn_mat[3][3] = {{1, dt, dt * dt / 2}, {0, 1, dt}, {0, 0, 1}};
    double dyn_jerk[3] = {dt * dt * dt / 6, dt * dt / 2, dt};
    double z_comp[3] = {1, 0, -h_com / grav};

    // Function to compute next x hat given the new jerk
    auto next_x = [&](const double x[3], const double x_jerk, double result[3]) {
        double temp[3];
        matVecMult3(dyn_mat, x, result);
        temp[0] = x_jerk * dyn_jerk[0];
        temp[1] = x_jerk * dyn_jerk[1];
        temp[2] = x_jerk * dyn_jerk[2];
        for (int i = 0; i < 3; ++i) {
            result[i] += temp[i];
        }
    };

    // Function to compute the new z dimension of the CoP given the current x hat
    auto compute_z = [&](const double x[3]) -> double {
        double sum = 0.0;
        for (int i = 0; i < 3; ++i) {
            sum += z_comp[i] * x[i];
        }
        return sum;
    };

    // Constraints/bounds
    std::vector<double> z_max(static_cast<int>(T / dt) + N, max_constraint);
    std::vector<double> z_min(static_cast<int>(T / dt) + N, min_constraint);
    for (int i = static_cast<int>(start_time / dt); i < static_cast<int>(start_time / dt + short_period); ++i) {
        z_max[i] = max_min_constraint;
    }
    for (int i = static_cast<int>(start_time / dt + short_period + period);
         i < static_cast<int>(start_time / dt + 2 * short_period + period); ++i) {
        z_max[i] = max_min_constraint;
    }
    for (int i = static_cast<int>(start_time / dt + 2 * short_period + 2 * period);
         i < static_cast<int>(start_time / dt + 3 * short_period + 2 * period); ++i) {
        z_max[i] = max_min_constraint;
    }
    for (int i = static_cast<int>(start_time / dt + short_period + diff_period);
         i < static_cast<int>(start_time / dt + 2 * short_period + diff_period); ++i) {
        z_min[i] = min_max_constraint;
    }
    for (int i = static_cast<int>(start_time / dt + 2 * short_period + period + diff_period);
         i < static_cast<int>(start_time / dt + 3 * short_period + period + diff_period); ++i) {
        z_min[i] = min_max_constraint;
    }
    for (int i = static_cast<int>(start_time / dt + 3 * short_period + 2 * period + diff_period);
         i < static_cast<int>(start_time / dt + 4 * short_period + 2 * period + diff_period); ++i) {
        z_min[i] = min_max_constraint;
    }

    real_t P_x[N * 3];
    for (int i = 0; i < N; ++i) {
        P_x[i * N + 0] = 1;
        P_x[i * N + 1] = dt * (i + 1);
        P_x[i * N + 2] = dt * dt * (i + 1) * (i + 1) / 2 - h_com / grav;
    }


    real_t P_u[N * N] = { 0.0 };
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            double coef = i - j;
            P_u[i * N + j] = (1 + 3 * coef + 3 * coef * coef) * dt * dt * dt / 6 - dt * h_com / grav;
        }
    }

    real_t P_u_mat[N][N] = {{0.0}};
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            double coef = i - j;
            P_u_mat[i][j] = (1 + 3 * coef + 3 * coef * coef) * dt * dt * dt / 6 - dt * h_com / grav;
        }
    }

    Eigen::Map<Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> P_u_x(P_u_mat[0], N, N);


    Eigen::MatrixXd P_u_x_inv = P_u_x.inverse();

    real_t P_u_inv[N * N] = { 0.0 };
 	for (int i = 0; i < P_u_x_inv.rows(); ++i) {
        for (int j = 0; j < P_u_x_inv.cols(); ++j) {
            P_u_inv[i * N + j] = P_u_x_inv(i, j);
        }
    }

    // Function to create eye matrix
    auto eyeMat = [&](double result[N*N]) {
        for (int i = 0; i < N; ++i) {
            result[i + i * N] = 1.0;
        }
    };

    // Function to matrix vector multiplication
    auto mat_vec_mul = [&](const double mat[N*3], const double vec[3], double result[N]) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < 3; ++j) {
                result[i] += mat[i * N + j] * vec[j];
            }
        }
    };

    // Function to matrix vector multiplication
    auto mat_vec_mul_n = [&](const double mat[N*N], const double vec[N], double result[N]) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                result[i] += mat[i * N + j] * vec[j];
            }
        }
    };

    // Function to create lower and upper constraint
    auto creat_constraint = [&](const int index, const double x_k[3], double lower[N], double upper[N]) {
        double temp[N] = { 0.0 };
        mat_vec_mul(P_x, x_k, temp);
        for (int i = 0; i < N; ++i) {
            upper[i] = z_max[index + i] - temp[i];
            lower[i] = z_min[index + i] - temp[i];
        }
    };


    // Function to create lower and upper constraint
    auto creat_constraint_ = [&](const int index, const double x_k[3], double lower[N], double upper[N]) {
        double temp[N] = { 0.0 };
        double upper_[N] = { 0.0 };
        double lower_[N] = { 0.0 };
        mat_vec_mul(P_x, x_k, temp);
        for (int i = 0; i < N; ++i) {
            upper_[i] = z_max[index + i] - temp[i];
            lower_[i] = z_min[index + i] - temp[i];
        }
		mat_vec_mul_n(P_u_inv, upper_, upper);
		mat_vec_mul_n(P_u_inv, lower_, lower);
		for (int i = 0; i < N; ++i){
			if (lower[i] > upper[i]) {
				double temp = lower[i];
				lower[i] = upper[i];
				upper[i] = temp;
			}
		}
    };

    /* Setup data of first QP. */
    real_t H[N*N] = { 0.0 };
    eyeMat(H);
    real_t A[N*N];
    std::copy(std::begin(P_u), std::end(P_u), std::begin(A));

    real_t x_k[3] = { 0.0 };

    real_t g[N] = { 0.0 };

	real_t lb[N] = { 0.0 };
    real_t ub[N] = { 0.0 };

    real_t lbA[N] = { 0.0 };
    real_t ubA[N] = { 0.0 };

    double cops[int(T / dt)] = { 0.0 };
    double jerks[int(T / dt)] = { 0.0 };
    for (int i = 0; i < int(T / dt); ++i) {
        std::cout << "Current step: " << i << std::endl;
        creat_constraint(i, x_k, lbA, ubA);
        creat_constraint_(i, x_k, lb, ub);

        /* Setting up QProblem object. */
        QProblem example( N,1 );

        Options options;
        options.printLevel = PL_NONE;
        example.setOptions( options );

        /* Solve first QP. */
        int_t nWSR = 10;
        example.init( H,g,A,lb,ub,lbA,ubA, nWSR );

        /* Get and print solution of first QP. */
        real_t xOpt[N];
        real_t yOpt[N + 1];
        example.getPrimalSolution( xOpt );
        example.getDualSolution( yOpt );
        real_t jerk = yOpt[0];

        real_t new_x[3];
        next_x(x_k, jerk, new_x);
        std::copy(std::begin(new_x), std::end(new_x), std::begin(x_k));

        cops[i] = compute_z(x_k);
        jerks[i] = jerk;
    }

    for (int i = 0; i < int(T / dt); ++i) {
        std::cout << cops[i] << " ";
    }
	std::cout << std::endl;
	std::cout << std::endl;
    for (int i = 0; i < int(T / dt); ++i) {
        std::cout << jerks[i] << " ";
    }
    return 0;
}
