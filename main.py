import numpy as np
import matplotlib.pyplot as plt
from plot import *
from gradient_descent import *


def linear_regression_mistake(dots: np.ndarray, func_arg: np.ndarray) -> float:
    return sum(abs(dot[1] - (func_arg[0] * dot[0] + func_arg[1])) for dot in dots)


def add_solution_to_plot(dots: np.ndarray, way: np.ndarray, show=True):
    # print(way)
    ans = way[-1]
    plt.plot(dots[:, 0], dots[:, 1], 'o')
    # print(ans)
    x_space = np.linspace(min(dots[:, 0]), max(dots[:, 0]), 30)
    plt.plot(x_space, [ans[0] * x + ans[1] for x in x_space])
    if show:
        plt.show()
    mistake = sum(abs(dot[1] - (ans[0] * dot[0] + ans[1])) for dot in dots)
    print(f"Mistake of solution is {mistake}")


def add_way_to_plot(dots: np.ndarray, way: np.ndarray, show=True):
    show_2arg_func(lambda func_arg: linear_regression_mistake(dots, func_arg), way, show=show)


def main():
    dots = np.array([
        [1, 1],
        [2, 7],
        [4, 2],
        [5, 6]
    ])
    # way = stochastic_gradient_descent(dots, 2, np.array([1.0, 0.0]), 0.01, constant_learning_rate())
    # way = running_average_stochastic_gradient_descent_constant(dots, 2, np.array([1.0, 0.0]), 0.01, 0.95)
    # way = nesterov_stochastic_gradient_descent_constant(dots, 2, np.array([1.0, 0.0]), 0.01, 0.95)
    # way = adagrad_stochastic_gradient_descent_constant(dots, 2, np.array([1.0, 0.0]), 0.3)
    # way = rmsprop_stochastic_gradient_descent_constant(dots, 2, np.array([1.0, 0.0]), 0.05, 0.99)
    way = adam_stochastic_gradient_descent_constant(dots, 2, np.array([1.0, 0.0]), 0.05, 0.9, 0.999)
    add_solution_to_plot(dots, way)
    add_way_to_plot(dots, way)


if __name__ == "__main__":
    main()
