<!-- 
Author(s): Shibaji Chakraborty

Disclaimer:
pyCHIPS is under the MIT license found in the root directory LICENSE.md 
Everyone is permitted to copy and distribute verbatim copies of this license 
document.

This version of the MIT Public License incorporates the terms
and conditions of MIT General Public License.
-->
# Checking CHIPS Assumptions

#### How do we know $x_{\tau}$ is a [Random Variable](https://en.wikipedia.org/wiki/Random_variable)?

We start our journey with the `Sun`, a dynamic celestial entity. By considering the Sun as a source of data, we treat it as a [Random Value Generator](https://en.wikipedia.org/wiki/Pseudorandom_number_generator). The surface processes of the Sun, akin to a [Stochastic Process] (https://en.wikipedia.org/wiki/Stochastic_process), infuse a sense of randomness into any measurement we take of its surface.

As we preprocess solar images, the pixel values on the resulting .fits file transform into a [Random Variable](https://en.wikipedia.org/wiki/Random_variable)/`RV`, ranging from 0 to $\infty$. This RV, resembling an exponential distribution (truncated), such as the `Gamma` distribution, implies that the parameter $\tau$ is also an `RV`.

Enter the [Logistic function](https://en.wikipedia.org/wiki/Logistic_function) $y_\tau$, a [Surjective function](https://en.wikipedia.org/wiki/Surjective_function) fitting onto $\tau$ and producing values in the [0-1] range. This, in turn, makes $y_\tau$ another `RV`. And behold, $x_\tau$ is simply the normalized version of $y_\tau$, maintaining its status as a [Random Variable](https://en.wikipedia.org/wiki/Random_variable).

Therefore, we deduce that $\tau$, $y_\tau$, and $x_\tau$ can be effectively modeled using Probability Density Functions ([PDFs](https://en.wikipedia.org/wiki/Probability_density_function)) of exponential distributions. This foundational understanding assures us that $x_{\tau}$ is not just a static value but a dynamic entity with inherent randomness. 

<figure markdown>
![Figure 07](../figures/Figure07.png)
<figcaption>Figure 01: Example showing PDFs that can fit through (a) $\tau$, (b) $y_\tau$, and (c) $x_\tau$ </figcaption>
</figure>

#### How do we know $\mathcal{F}(I>I_{th},x_{\tau_th})$ is an [Random Variable](https://en.wikipedia.org/wiki/Random_variable) and can be described using [PDF](https://en.wikipedia.org/wiki/Probability_density_function)?

To show that the integration of a [PDF](https://en.wikipedia.org/wiki/Probability_density_function) results in a uniform distribution, we use the Cumulative Distribution Function ([CDF](https://en.wikipedia.org/wiki/Cumulative_distribution_function)). The CDF is the integral of the PDF.

Let $f_x$ be the PDF of a [Random Variable](https://en.wikipedia.org/wiki/Random_variable)/`RV` `Z` and $F_X$ be the corresponding CDF, $F_X(x) = \int_{-\infty}^x f_x dx$. We understand $F_X$ values lies between [0-1] and monotonically increasing. We define $Z=F_X(X)$, then $F_Z(X) = Porb(F_X(X)\leq x)= Prob(X\leq F^{-1}_X(x))= F_X(F^{-1}_X(x))=x$.

The derivative of $F_Z$ with respect to $x$ is a constant is constant so Z is uniformly distributed with a PDF $\mathcal{U}(0,1)$.