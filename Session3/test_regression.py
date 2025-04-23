import numpy as np
import matplotlib.pyplot as plt

ys = np.array([130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167], dtype=float)
xs = np.array([148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173], dtype=float)


def linear(params, x):
    a, b = params
    return a * x + b

def cost(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def calculate_new_a(a, b, xs, ys, learning_rate):
    y_pred = linear([a, b], xs)
    gradient = np.mean(2 * (y_pred - ys) * xs)
    return a - learning_rate * gradient

def calculate_new_b(a, b, xs, ys, learning_rate):
    y_pred = linear([a, b], xs)
    gradient = np.mean(2 * (y_pred - ys))
    return b - learning_rate * gradient

def main():
    a = 0.555
    b = 94.585026
    learning_rate = 0.000001
    epochs = 1000
    
    print("Initial predictions:")
    for x, y in zip(xs, ys):
        estimated_y = linear([a, b], x)
        print(f"correct answer: {y}")
        print(f"estimated: {estimated_y}")
        print("******")
    
    print("\nTraining model...")
    for epoch in range(epochs):
        y_pred = linear([a, b], xs)
        current_cost = cost(ys, y_pred)
        print(f"Epoch {epoch+1}, Cost: {current_cost:.4f}, Parameters: a={a:.6f}, b={b:.6f}")
        
        new_a = calculate_new_a(a, b, xs, ys, learning_rate)
        new_b = calculate_new_b(a, b, xs, ys, learning_rate)
        a, b = new_a, new_b

    print(f"a : {a:.6f}, b : {b:.6f}")

    plt.scatter(xs, ys, color='blue', label='Data points')
    plt.xlabel('xs')
    plt.ylabel('ys')
    plt.title('Scatter plot of xs vs ys')
    plt.legend()
    plt.plot(xs, linear([a, b], xs), color='red', label='Fitted line')
    plt.show()
    


if __name__ == "__main__":
    main()
