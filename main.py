import numpy as np
import matplotlib.pyplot as plt

import func_utils
from plot import *
from gradient_descent import *


def linear_regression_mistake(dots: np.ndarray, func_arg: np.ndarray) -> float:
    return sum(abs(dot[1] - (func_arg[0] * dot[0] + func_arg[1])) for dot in dots)


def add_solution_to_plot(dots: np.ndarray, way: np.ndarray, show=True):
    # print(way)
    ans = way[-1]
    curve = func_utils.curve(ans)
    func = func_utils.func(curve)
    plt.plot(dots[:, 0], dots[:, 1], 'o')
    # print(ans)
    x_space = np.linspace(min(dots[:, 0]), max(dots[:, 0]), 30)
    plt.plot(x_space, [curve(x) for x in x_space])
    if show:
        plt.show()
    mistake = func_utils.mistake(func, dots)
    print(f"Mistake of solution is {mistake}")


def add_way_to_plot(dots: np.ndarray, way: np.ndarray, show=True):
    show_2arg_func(lambda func_arg: linear_regression_mistake(dots, func_arg), way, show=show)


def main():
    # dots = np.array([[i, i ** 2] for i in range(-5, 6)])
    dots = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3]
    ])
    # dots = np.array([[i, math.sin(i)] for i in range(-6, 6)])
    # dots = np.array([
    #     [-1, 7],
    #     [0, 1],
    #     [1, -1],
    #     [2, -5],
    #     [3, 7]
    # ])
    # way1 = stochastic_gradient_descent(dots, 2, np.array([1.0, 0.0]), 0.02, constant_learning_rate())
    # way2 = stochastic_factory(2, 0.02, constant_learning_rate(), func_utils.grad_func,
    #                           lambda prev, lr, gr: prev - lr * gr)(dots, np.array([1.0, 0.0]))
    # way1 = running_average_stochastic_gradient_descent_constant(dots, 2, np.array([1.0, 0.0]), 0.01, 0.95)
    # way2 = stochastic_factory(2, 0.01, constant_learning_rate(),
    #                           func_utils.grad_func, running_average(0.95))(dots, np.array([1.0, 0.0]))
    # way1 = nesterov_stochastic_gradient_descent_constant(dots, 2, np.array([1.0, 0.0]), 0.01, 0.95)
    # way2 = stochastic_factory(2, 0.01, constant_learning_rate(),
    #                           *nesterov_mod(0.95, func_utils.grad_func))(dots, np.array([1.0, 0.0]))
    # way1 = adagrad_stochastic_gradient_descent_constant(dots, 2, np.array([1.0, 0.0]), 0.3)
    # way2 = stochastic_factory(2, 0.3, constant_learning_rate(),
    #                           func_utils.grad_func, adagrad_mod())(dots, np.array([1.0, 0.0]))
    # way1 = rmsprop_stochastic_gradient_descent_constant(dots, 2, np.array([1.0, 0.0]), 0.05, 0.95)
    # way2 = stochastic_factory(2, 0.05, constant_learning_rate(),
    #                           func_utils.grad_func, rmsprop_mod(0.95))(dots, np.array([1.0, 0.0]))
    # way1 = adam_stochastic_gradient_descent_constant(dots, 2, np.array([1.0, 0.0]), 0.05, 0.9, 0.999)
    # way2 = stochastic_factory(2, 0.05, constant_learning_rate(),
    #                           func_utils.grad_func, adam_mod(0.9, 0.999))(dots, np.array([1.0, 0.0]))
    # way1 = polynomial_stochastic_gradient_descent(dots, 2, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), 0.00005, l1=1, l2=-3, elastic=4)
    convergred, way = stochastic_factory(4, 0.007, linear_learning_rate_scheduler(0.99, 100),
                                         func_utils.grad_func,
                                         default_new_args, epoch_limit=2000) \
        (dots, np.array([0.0, 0.0]))
    print(convergred)
    add_solution_to_plot(dots, way)
    # add_solution_to_plot(dots, way2)

    # print(func_utils.mistake(func_utils.func(func_utils.curve(np.array([1.0, -1.0, 1.0, -3.0, 1.0]))), dots))
    # curve = func_utils.curve(np.array([1.0, -1.0, 1.0, -3.0, 1.0]))
    # func = func_utils.func(curve)
    # grad_func = func_utils.grad_func(np.array([1.0, -1.0, 1.0, -3.0, 1.0]))
    # for dot in dots:
    #     func_val = func(dot[0], dot[1])
    #     grad_val = grad_func(dot[0], dot[1])
    #     print(f"func({dot}) == {func_val}")
    #     print(f"grad({dot}) == {grad_val}")

    # add_way_to_plot(dots, way)


if __name__ == "__main__":
    main()
