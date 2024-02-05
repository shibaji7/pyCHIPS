<!-- 
Author(s): Shibaji Chakraborty

Disclaimer:
pyCHIPS is under the MIT license found in the root directory LICENSE.md 
Everyone is permitted to copy and distribute verbatim copies of this license 
document.

This version of the MIT Public License incorporates the terms
and conditions of MIT General Public License.
-->

#### How do we know $x_{\tau}$ is an [Random Variable](https://en.wikipedia.org/wiki/Random_variable)?
We assume the source of the dataset `The Sun` is a random value generator. The processes in the surface of the Sun can be modelded using a [Stochastic Process](https://en.wikipedia.org/wiki/Stochastic_process). Hence, any measurement of the Sun's surface can be assumed as a [Random Variable](https://en.wikipedia.org/wiki/Random_variable)/`RV`. After preprocessing the solar image, the values of the pixels on `.fits` file rainging between [0-$\infty$), which is an `RV` and can be modeled as any exponetial distribution (truncated), e.g., `Gamma`. Hence, the parameter `\tau` is also an `RV`, this suggests it can also be modeled as any exponetial distribution. Note that, $y_\tau$ is a [Logistic function](https://en.wikipedia.org/wiki/Logistic_function) fit (a [Surjective function](https://en.wikipedia.org/wiki/Surjective_function)) on $\tau$, with values ranging between [0-1]. Hence, $y_\tau$ is also an `RV` and $x_\tau$ is nothing but normalized $y_\tau$, and is an `RV`. Hence $\tau$, $y_\tau$ and $x_\tau$ can be modeled using PDFs of exponetial distribution.

<figure markdown>
![Figure 07](../figures/Figure07.png)
<figcaption>Figure 01: Example showing PDFs that can fit through (a) $\tau$, (b) $y_\tau$, and (c) $x_\tau$ </figcaption>
</figure>

#### How do we know $\mathcal{F}(I>I_{th},x_{\tau_th})$ is an [Random Variable](https://en.wikipedia.org/wiki/Random_variable) and can be described using [PDF](https://en.wikipedia.org/wiki/Probability_density_function)?