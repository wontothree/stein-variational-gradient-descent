# Stein Variational Gradient Descent

![](./imgs/algorithm.png)

# Simulation

target distribution

```py
A = np.array([[0.2260, 0.1652], [0.1652, 0.6779]])
mu = np.array([-4, 3])
```

initial distribution

```py
x0 = np.random.normal(7, 7, [100, 2])
```

![svgd_evolution](https://github.com/user-attachments/assets/bb98cef4-2fc5-4167-b5bd-3b8cfb794a7b)

```bash
ground truth:  [-4  3]
svgd:  [-3.90359892  3.03488806]
```
