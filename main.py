import numpy as np
import matplotlib.pyplot as plt

from gradient_descent import *


def main():
    dots = np.array([
        [0, 2],
        [1, 1],
        [2, 2],
        [3, 1],
        [4, 2],
        [5, 1],
        [6, 2],
        [7, 1]
    ])
    way = stochastic_gradient_descent_constant(dots, 8, np.array([1.0, 3.0]), 0.01)
    print(way)
    ans = way[-1]
    plt.plot(dots[:, 0], dots[:, 1], 'o')
    print(ans)
    x_space = np.linspace(min(dots[:, 0]), max(dots[:, 0]), 30)
    plt.plot(x_space, [ans[0] * x + ans[1] for x in x_space])
    plt.show()
    mistake = sum(abs(dot[1] - (ans[0] * dot[0] + ans[1])) for dot in dots)
    print(f"Mistake of solution is {mistake}")


if __name__ == "__main__":
    main()
