Gradient descent is a method for unconstrained mathematical optimization. It is a first-order iterative algorithm for minimizing a differentiable multivariate function.

The idea is to take repeated steps in the opposite direction of the gradient (or approximate gradient) of the function at the current point, because this is the direction of steepest descent. Conversely, stepping in the direction of the gradient will lead to a trajectory that maximizes that function; the procedure is then known as gradient ascent.
Gradient descent should not be confused with local search algorithms, although both are iterative methods for optimization.
Gradient descent is particularly useful in machine learning and artificial intelligence for minimizing the cost or loss function. 
Gradient descent is generally attributed to Augustin-Louis Cauchy, who first suggested it in 1847. Jacques Hadamard independently proposed a similar method in 1907. Its convergence properties for non-linear optimization problems were first studied by Haskell Curry in 1944, with the method becoming increasingly well-studied and used in the following decades.
A simple extension of gradient descent, stochastic gradient descent, serves as the most basic algorithm used for training most deep networks today.


== Description ==

Gradient descent is based on the observation that if the multi-variable function f(𝐱) is defined and differentiable in a neighborhood of a point 𝐚, then f(𝐱) decreases fastest if one goes from 𝐚 in the direction of the negative gradient of f at 𝐚,i.e.,−∇f(𝐚). It follows that, if

𝐚n+1=𝐚n−η∇f(𝐚n)
for a small enough step size or learning rate η∈ℝ+, then  f(𝐚𝐧)≥f(𝐚𝐧+𝟏). In other words, the term η∇f(𝐚) is subtracted from 𝐚 because we want to move against the gradient, toward the local minimum. With this observation in mind, one starts with a guess 𝐱0 for a local minimum of f, and considers the sequence 𝐱0,𝐱1,𝐱2,… such that

𝐱n+1=𝐱n−ηn∇f(𝐱n), n≥0.
We have a monotonic sequence

f(𝐱0)≥f(𝐱1)≥f(𝐱2)≥⋯,
so the sequence (𝐱n) converges to the desired local minimum. Note that the value of the step size η is allowed to change at every iteration. 
It is possible to guarantee the convergence to a local minimum under certain assumptions on the function f (for example, f convex and ∇f Lipschitz) and particular choices of η. Those include the sequence
ηn=|(𝐱n−𝐱n−1)⊤[∇f(𝐱n)−∇f(𝐱n−1)]|‖∇f(𝐱n)−∇f(𝐱n−1)‖2
as in the Barzilai-Borwein method, or a sequence ηn satisfying the Wolfe conditions (which can be found by using line search). When the function f is convex, all local minima are also global minima, so in this case gradient descent can converge to the global solution.
