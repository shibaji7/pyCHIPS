<!-- 
Author(s): Shibaji Chakraborty

Disclaimer:
pyCHIPS is under the MIT license found in the root directory LICENSE.md 
Everyone is permitted to copy and distribute verbatim copies of this license 
document.

This version of the MIT Public License incorporates the terms
and conditions of MIT General Public License.
-->

# Unveiling the Intricacies of CHIPS Algorithm: A Step-by-Step Guide with a Case Study
We delve into the intricacies of the **CHIPS** algorithm. This page offers a comprehensive guide alongside an illustrative example using images captured on 30 May 2018. Despite the array of semi-automatic and automatic schemes available for Coronal Hole Boundary (CHB) detection [@Krista2009;@Garton2018;@Illarionov2018], CHIPS stands out due to its unique selling proposition. What sets CHIPS apart is its ability to provide a **probability** [$\theta$] for the identified Coronal Hole (CH) and its associated CHB. This distinctive feature equips researchers with a quantifiable measure of certainty regarding the detection mechanism. Join us as we navigate through the sequential steps of the CHIPS algorithm, gaining insights into its functionality and the invaluable certainty it brings to CH and CHB identification.

## The CHIPS Pipeline
<figure markdown>
![Figure 01](../figures/Figure01.png)
<figcaption>Figure 01: SDO AIA full-disk images of the Sun in three wavebands, (a) 17.1 nm, (b) 19.3 nm, and (c) 21.1 nm taken on 30 May 2018. Example coronal hole located near the central meridian in all three panels.</figcaption>
</figure>

The algorithm is developed to run on full-disk and synoptic images. The basic processing units [**U**] are shared between the different image types. These processing units exploit knowledge from solar physics to segment CHs with different threshold values on the solar disk and then estimate CHBs. This includes the concept that boundaries are best visible in 17.1 nm, 19.3 nm, and 21.1 nm wavebands of AIA images, and the image intensity within CHs are less than their neighboring pixels (shown in Figure 01). Four major stages of the scheme are described below, Figure 02 presents flow diagram of the algorithm, and Figure 03 presents the corresponding outputs from each stage. The algorithm can take either of the 4k resolution images presented in Figure 01 or synoptic maps as an input.

![Model framework architecture for CHIPS showing the component modules and their interconnection.](../figures/Figure02.png)



\bibliography