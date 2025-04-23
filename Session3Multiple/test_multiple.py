import numpy as np
import matplotlib.pyplot as plt

# Données
xs = [
    [93, 230, 250, 260, 119, 183, 151, 192, 263, 185],
    [150, 311, 182, 245, 152, 162, 99, 184, 115, 105]
]
ys = [123, 290, 230, 261, 140, 173, 133, 179, 210, 181]

X = np.array(xs).T        
y = np.array(ys, dtype=float)

def predict(X, w, b):
    return X.dot(w) + b

def cost(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def gradients(X, y_true, y_pred):
    n = len(y_true)
    error = y_pred - y_true
    dw = (2/n) * X.T.dot(error)
    db = (2/n) * np.sum(error)
    return dw, db

def main():
    w = np.zeros(X.shape[1])   
    b = 0.0
    lr = 1e-6
    epochs = 1000

    print(X.shape)
    print(y.shape)

    first_pred = predict(X, w, b)
    print("Prédictions initiales :" , first_pred)

    loss_initial = cost(y, first_pred)
    print("Coût initial :", loss_initial)

    gradients_initial = gradients(X, y, first_pred)
    print("Gradients initiaux :", gradients_initial)

    for epoch in range(epochs):
        y_pred = predict(X, w, b)
        J = cost(y, y_pred)
        dw, db = gradients(X, y, y_pred)
        w -= lr * dw
        b -= lr * db
        if epoch % 100 == 0:
            print(f"Epoch {epoch:5d} | Cost: {J:.4f} | w: {w} | b: {b:.4f}")

    print("\nParamètres finaux :")
    print(f"w = {w}, b = {b:.4f}")


    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X[:,0], X[:,1], y, c='blue', label='Données réelles')
    # x1g = np.linspace(X[:,0].min(), X[:,0].max(), 10)
    # x2g = np.linspace(X[:,1].min(), X[:,1].max(), 10)
    # x1m, x2m = np.meshgrid(x1g, x2g)
    # ym = w[0]*x1m + w[1]*x2m + b
    # ax.plot_surface(x1m, x2m, ym, alpha=0.5, color='red')
    # ax.set_xlabel('Feature 1')
    # ax.set_ylabel('Feature 2')
    # ax.set_zlabel('Target (ys)')
    # ax.legend()
    # plt.show()

if __name__ == "__main__":
    main()