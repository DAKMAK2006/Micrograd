# Micrograd from scratch

This is my minimal implementation of an autodiff engine, done from scratch in python while following Andrej Karpathy’s tutorial.

I’ve implemented:
- A basic `Value` class that tracks data and gradients
- Operator overloading for building computation graphs
- Manual training loop for a tiny neural network
- Backpropagation from scratch

This project helped me understand:
- How autograd works internally
- What backward passes actually mean
- How neural networks are *actually* trained under the hood
- How to debug from gradients to final outputs

---

## Files
- `micrograd_from_scratch.ipynb`: Contains everything — code, comments, and training loop
- Not using any ML library here — it’s pure Python and math

---

## Output Sample
After a few iterations, you’ll see printed loss values decreasing like:

```
0 21.142276445052506
1 16.830653144768175
...
```

---

## Why I Did This
It was the first time that I did something entirly from scratchin neural networks cause untill now I had implemented these neural networks but using some library here and there and that also under a certified course so files were pre-written. It was a really good learning experience.

---

DO check out the notebook as it contains a bit deeper intuition on the actual code. Suggestions are welcome!
