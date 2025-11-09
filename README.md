# flam-rd-parametric-curve-fit

This MATLAB project was developed to invert nonlinear parametric curve equations and accurately recover unknown parameters θ, M and X based solely on observed (x,y) points. The optimization techniques used in this project combine genetic algorithms (GA) with sensitivity-guided range compression, inspired by the signal processing models developed for MARSIS—a NASA radar sounder mission on Mars. During my internship at ISRO, I was working on a signal processing algorithm based on Bayesian inference and genetic algorithms for MARSIS data inversion. This project adapts similar approaches for accurate parameter recovery in nonlinear parametric curve fitting.

The workflow started by using basic GA fitting on unordered and sorted data to explore how data ordering impacts fit quality. Sorting the data was done based on the assumption that the parameter t varied linearly: t<sub>all</sub> = linspace (6,60,length(x<sub>obs</sub>)). Sorting X and reordering Y accordingly was done under the hope that sorting preserves point correspondences.

However, this approach technically "shouldn't" work because it assumes the sorted X order corresponds exactly to a linear progression in t. In reality, the actual t values linked with the observed points are unknown and need not follow any linear order. Miraculously, for this specific curve shape, sorting by X approximately matched the true ordering of t, enabling reasonable parameter recovery at this stage.

To overcome the limitations of this assumption, the project then transitioned to a more robust point-to-curve distance minimization method. This calculates the minimum distance from each observed point to the parametric curve, without assuming a fixed t order. This significantly improved fit accuracy (lower L1 distance), though at the cost of slower computation.

To speed up the inversion with comparable accuracy, Jacobian-based sensitivity analysis was introduced. This powerful method analyzes parameter sensitivities and narrows the search space for genetic algorithm optimization. Using this guided GA approach with ultra-tight parameter ranges, the L1 misfit was reduced effectively to about 11, producing high-quality parameter estimates much faster than the pure point-to-curve approach.

This stepwise development from simple sorted data GA to robust point-to-curve matching and then guided sensitivity optimization illustrates the practical trade-offs of accuracy, assumptions, and computation time in nonlinear parametric inversion.

## Optimal Parameters <br>
θ = 29.999947 deg or 0.523597 rad <br>
M = 0.02999992 <br>
X = 54.999945 <br>
L1 = 4.849061 <br>
\left(\left(t*\cos(0.523597)-e^{0.02999992\left|t\right|}\cdot\sin(0.3t)\sin(0.523597)\ +54.999945\right)\ ,\left(42+\ t*\sin(0.523597)+e^{0.02999992\left|t\right|}\cdot\sin(0.3t)\cos(0.523597)\right)\right)  <br>

## FILE STRUCTURE
```
├── data_exploration.m                # Data visualization
├── forward_model.m                   # Parametric curve definition
├── comp_visualization.m              # Model component analysis
│
├── inversion_models/                 # Parameter estimation methods
│   ├── basic_GA_fitting/             # Initial GA approaches
│   │   ├── non_sorted_GA.m           # Raw data (L1 ≈ 37865)
│   │   └── sorted_GA.m               # Sorted data (L1 ≈ 453)
│   ├── point_to_curve/               # Robust distance minimization
│   │   └── point_to_curve.m          # Best accuracy (L1 ≈ 4.8)
│   ├── guided_fitting/               # Sensitivity-guided optimization
│   │   ├── Jacobian_analysis.m       # Parameter sensitivity
│   │   └── guided_GA.m               # Guided GA (L1 ≈ 15)
│   └── ultratight_guided_fitting/    # Final optimization
│       ├── ultratight_Jacobian_analysis.m
│       └── ultratight_guided_GA.m    # Final (L1 ≈ 11)
│
└── xy_data.csv                    # Observed data points
```
## RUNNING THE CODE
1. Clone the repository and ensure xy_data.csv is present.
2. Visualize data with data_exploration.m.
3. Explore the forward model using forward_model.m.
4. Run inversion scripts from basic GA fitting to ultra-tight guided fitting progressively in the inversion_models folder to see improvements.
5. Output visualizations of fit quality and parameter diagnostics are saved in the figures folder.

## CITATION 
This project adapts methodology and concepts from the paper:
> Dielectric properties of the Martian south polar layered deposits: MARSIS data inversion using Bayesian inference and genetic algorithm, Zhenfei Zhang et al., Journal of Geophysical Research, 2008.

It leverages Bayesian inference ideas and GA strategies for accurate inversion in complex nonlinear forward models.
