import numpy as np

def newton_method(f, df, d2f, x0, tol=1e-6, max_iter=100):
    """
    Minimize a function using Newton's method.
    Args:
        f: function to minimize
        df: first derivative of f
        d2f: second derivative of f
        x0: initial guess
        tol: tolerance for stopping
        max_iter: maximum number of iterations
    Returns:
        x_min: the estimated location of the minimum
        history: list of iterates
    """
    x = x0
    history = [x]
    iterations = 0
    for i in range(max_iter):
        grad = df(x)
        hess = d2f(x)
        if abs(hess) < 1e-12:
            print("Zero or near-zero second derivative. Stopping.")
            break
        x_new = x - grad / hess
        history.append(x_new)
        iterations += 1
        if abs(x_new - x) < tol:
            break
        x = x_new
    return x, history, iterations

if __name__ == "__main__":
    # Example: Minimize f(x) = (x-2)^4 + 1
    f = lambda x: (x-2)**4 + 1
    df = lambda x: 4*(x-2)**3
    d2f = lambda x: 12*(x-2)**2
    x0 = 0.0
    x_min, hist, num_iter = newton_method(f, df, d2f, x0)
    print(f"Minimum at x = {x_min}")
    print(f"Function value at minimum: f(x) = {f(x_min)}")
    print(f"Iterations: {num_iter}")
    print(f"History: {hist}")

    # Plot function value after every iteration
    import matplotlib.pyplot as plt
    f_values = [f(x) for x in hist]
    plt.figure(figsize=(8, 4))
    plt.plot(f_values, marker='o')
    plt.title('Function Value at Each Newton Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
