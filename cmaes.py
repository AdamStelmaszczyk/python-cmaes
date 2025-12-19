import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


def make_plot(fitnessfct: Callable[[np.ndarray], float]) -> plt.Artist:
    """
    Makes a plot for visualizing the fitness landscape and population evolution.

    :param fitnessfct: The objective function.
    :return: A matplotlib plot.
    """
    # Create the figure and axis for plotting
    fig, ax = plt.subplots()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title("Population evolution")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    # Generate meshgrid for plotting the fitness landscape
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([fitnessfct(np.array([x, y])) for x, y in zip(X.flatten(), Y.flatten())]).reshape(X.shape)

    # Plot the fitness landscape as a contour plot
    contour = ax.contourf(X, Y, Z, 50, cmap='viridis')
    fig.colorbar(contour, ax=ax, label='fitness')

    plot = ax.scatter([], [], color='r', s=10)
    return plot


def cmaes(
        fitnessfct: Callable[[np.ndarray], float],
        xmean: np.ndarray,
        sigma: float,
        stopfitness: float,
        stopeval: int,
        visualize: bool
) -> np.ndarray:
    """
    CMA-ES: Evolution Strategy with Covariance Matrix Adaptation for nonlinear function minimization.
    Strictly following the Matlab code from "The CMA Evolution Strategy: A Tutorial" by Nikolaus Hansen:
    https://arxiv.org/pdf/1604.00772

    :param fitnessfct: The objective function to minimize.
    :param xmean: Initial point.
    :param sigma: Initial step size.
    :param stopfitness: Stop if fitness < stopfitness.
    :param stopeval: Stop after stopeval number of function evaluations.
    :param visualize: Show the visualization?
    :return: Best point found.
    """
    # ------------------------------ Initialization ------------------------------ #
    N = xmean.size

    # Strategy parameter setting: Selection
    lambda_ = 4 + math.floor(3 * np.log(N))  # population size, offspring number
    print("lambda", lambda_)
    mu = lambda_ // 2  # number of parents/points for recombination
    print("mu", mu)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))  # muXone recombination weights
    weights = weights / np.sum(weights)  # normalize recombination weights array
    print("weights", weights)
    mueff = 1 / sum(weights ** 2)  # variance-effective size of mu
    print("mueff", mueff)

    # Strategy parameter setting: Adaptation
    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)  # time constant for cumulation for C
    print("cc", cc)
    cs = (mueff + 2) / (N + mueff + 5)  # t-const for cumulation for sigma control
    print("cs", cs)
    c1 = 2 / ((N + 1.3) ** 2 + mueff)  # learning rate for rank-1 update
    print("c1", c1)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + 2 * mueff / 2))  # for rank-mu update
    print("cmu", cmu)
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs  # damping for sigma
    print("damps", damps)

    # Initialize dynamic (internal) strategy parameters and constants
    pc = np.zeros(N)  # evolution path for C
    ps = np.zeros(N)  # evolution path for sigma
    B = np.eye(N)  # B defines the coordinate system (eigenvectors)
    D = np.eye(N)  # diagonal matrix D defines the scaling (eigenvalues)
    C = np.eye(N)  # covariance matrix
    eigenval = 0
    chiN = np.sqrt(N * (1 - 1 / (4 * N) + 1 / (21 * N ** 2)))  # expectation of ||N(0,I)|| == norm(randn(N,1))
    print("chiN", chiN)
    print()

    plot = make_plot(fitnessfct)
    # ------------------------------ Generation loop ------------------------------ #
    counteval = 0
    while counteval < stopeval:
        # Generate and evaluate lambda offsprings

        # Version matching the Matlab code:
        arfitness = np.zeros(lambda_)
        arx = np.zeros(shape=(lambda_, N))
        arz = np.zeros(shape=(lambda_, N))
        for k in range(lambda_):
            arz[k] = np.random.randn(N)  # standard normally distributed vector
            arx[k] = xmean + sigma * (B @ D @ arz[k])  # add mutation
            arfitness[k] = fitnessfct(arx[k])  # objective function call
            counteval += 1

        # Vectorized version (same results, but faster):
        # arz = np.random.randn(lambda_, N)  # Generate all offspring in one go
        # arx = xmean + sigma * (arz @ D) @ B.T  # Apply mutation to all at once
        # arfitness = np.apply_along_axis(fitnessfct, 1, arx)  # Evaluate fitness for all - this can be parallelized on many cores
        # counteval += lambda_  # Update evaluation counter

        if visualize:
            plot.set_offsets(arx)  # update positions of the population
            plt.pause(0.01)  # pause to update the plot

        # Sort by fitnesss and compute weighted mean into xmean
        arindex = np.argsort(arfitness)  # minimization
        arfitness = arfitness[arindex]
        best_x = arx[arindex][:mu]
        xmean = best_x.T @ weights  # recombination
        best_z = arz[arindex[:mu]]
        zmean = best_z.T @ weights  # == Dˆ-1*B’*(xmean-xold)/sigma
        # Cumulation: update evolution paths
        ps = (1 - cs) * ps + (np.sqrt(cs * (2 - cs) * mueff)) * (B @ zmean)
        # Heaviside (on/off) function
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / lambda_)) / chiN < 1.4 + 2 / (N + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (B @ D @ zmean)

        # Adapt covariance matrix
        # TODO: I followed the Matlab code, but equation 47 is different, hsig is in the old_matrix, not rank_one
        old_matrix = (1 - c1 - cmu) * C
        rank_one = c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
        rank_mu = cmu * (B @ D @ best_z.T) @ np.diag(weights) @ (B @ D @ best_z.T).T
        C = old_matrix + rank_one + rank_mu

        # Adapt step-size sigma
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        # Update B and D from C
        if counteval - eigenval > lambda_ / (c1 + cmu) / N / 10:
            eigenval = counteval
            C = np.triu(C) + np.triu(C, k=1).T  # enforce symmetry
            D, B = np.linalg.eigh(C)  # eigen decomposition, B == normalized eigenvectors
            D = np.sqrt(np.diag(D))  # D contains standard deviations now

        # Break, if fitness is good enough
        if arfitness[0] <= stopfitness:
            break

        # Escape flat fitness, or better terminate?
        if arfitness[0] == arfitness[math.ceil(0.7 * lambda_)]:
            sigma = sigma * math.exp(0.2 + cs / damps)
            print("warning: flat fitness, consider reformulating the objective")

        print(f"{counteval} : {arfitness[0]}")

    if visualize:
        plt.show()  # show the final plot and block
    return xmean


def sphere(x: np.ndarray) -> float:
    '''
    Sphere function. Global minimum (center) at zero in all dimensions.
    :param x: A point represented as a NumPy array.
    :return: Euclidean distance from the given point to the center of the sphere.
    '''
    return np.sum(x ** 2)


def ellipsoid(x: np.ndarray) -> float:
    """
    Ellipsoid function.

    :param x: A point represented as a NumPy array.
    :return: The fitness value of the ellipsoid function at the given point.
    """
    N = x.size
    if N < 2:
        raise ValueError("Dimension must be greater than one")
    condition_number = 1e6
    exponents = np.arange(0, N) / (N - 1)
    scaling_factors = condition_number ** exponents
    return np.sum(scaling_factors * x ** 2)


def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin function. A multimodal function commonly used for testing optimization algorithms.

    :param x: A point represented as a NumPy array.
    :return: The fitness value of the Rastrigin function at the given point.
    """
    A = 10
    N = x.size
    return A * N + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CMA-ES: Evolution Strategy with Covariance Matrix Adaptation")
    parser.add_argument('-f', '--function', choices=['sphere', 'ellipsoid', 'rastrigin'], default='rastrigin',
                        help="The objective function (default: rastrigin)")
    parser.add_argument('-n', type=int, default=2, help="N, number of dimensions (default: 2)")
    parser.add_argument('-s', '--seed', type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument('-v', '--visualize', action="store_true", help="Enable visualization (default: False)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    N = args.n
    fitness_function = locals()[args.function]
    best_point = cmaes(
        fitnessfct=fitness_function,
        xmean=np.random.rand(N),
        sigma=0.5,
        stopfitness=1e-5,
        stopeval=1000 * N ** 2,
        visualize=args.visualize,
    )
    print("best_point", best_point)
    print("fitness", fitness_function(best_point))
