<!-- 
Author(s): Shibaji Chakraborty

Disclaimer:
pyCHIPS is under the MIT license found in the root directory LICENSE.md 
Everyone is permitted to copy and distribute verbatim copies of this license 
document.

This version of the MIT Public License incorporates the terms
and conditions of MIT General Public License.
-->

# Example Event: 13:00 UT, 23 June 2018
Explore the implementation of `pyCHIPS` through this illustrative Python code, which serves as a valuable resource for detecting Coronal Holes (CHs) and assessing their associated probabilities. The code provides a step-by-step demonstration of how to leverage `pyCHIPS` functionalities to identify CHs in solar imagery data. This hands-on example offers insights into the practical application of `pyCHIPS` for space weather research, enabling users to gain a deeper understanding of the CH detection process and the computation of associated probabilities. Whether you are a seasoned Python developer or new to space weather analysis, this code snippet offers a clear and insightful walkthrough, making it an excellent reference for those interested in utilizing `pyCHIPS` for CH detection and probability assessment in solar physics.

```
# Import nessesory modules
from fetch import RegisterAIA
from chips import Chips
from plots import ImagePalette, Annotation
```
Load a high-resolution 4K image from the Atmospheric Imaging Assembly (AIA) at the 19.3 nm wavelength into your analysis using the provided Python code. Simply load the image, ensuring that the `apply_psf` parameter is set to `true` if you wish to invoke the Point Spread Function (PSF) specific to the AIA. It's crucial to note that enabling the PSF may lead to a substantial increase in time complexity due to the detailed computational processes involved. This code snippet serves as a valuable guide for researchers and developers seeking to integrate AIA imagery, offering flexibility in resolution and the option to apply the PSF for enhanced accuracy in their solar physics analyses. Whether you are an experienced user or new to AIA image processing, this code provides clear instructions for efficient implementation.

```
aia193 = RegisterAIA(
    self.date, [193], [4096], 
    apply_psf=False
)
```

