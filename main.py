import numpy as np
import matplotlib.pyplot as plt
from plot import *
from gradient_descent import *


def linear_regression_mistake(dots: np.ndarray, func_arg: np.ndarray) -> float:
    return sum(abs(dot[1] - (func_arg[0] * dot[0] + func_arg[1])) for dot in dots)


def main():
    dots = np.array([
        [1, 1],
        [2, 7],
        [4, 2],
        [5, 6]
    ])
    way = stochastic_gradient_descent(dots, 2, np.array([1.0, 0.0]), 0.05,
                                      exponent_learning_rate_scheduler(0.3))
    print(way)
    ans = way[-1]
    plt.plot(dots[:, 0], dots[:, 1], 'o')
    print(ans)
    x_space = np.linspace(min(dots[:, 0]), max(dots[:, 0]), 30)
    plt.plot(x_space, [ans[0] * x + ans[1] for x in x_space])
    plt.show()
    mistake = sum(abs(dot[1] - (ans[0] * dot[0] + ans[1])) for dot in dots)
    print(f"Mistake of solution is {mistake}")

    # show the way of gradient descent
    show_2arg_func(lambda func_arg: linear_regression_mistake(dots, func_arg), way)


if __name__ == "__main__":
    main()
