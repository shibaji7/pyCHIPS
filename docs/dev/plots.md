<!-- 
Author(s): Shibaji Chakraborty

Disclaimer:
pyCHIPS is under the MIT license found in the root directory LICENSE.md 
Everyone is permitted to copy and distribute verbatim copies of this license 
document.

This version of the MIT Public License incorporates the terms
and conditions of MIT General Public License.
-->

::: chips.plots.Annotation
    handler: python
    options:
    options:
      show_root_heading: true
      show_source: false

::: chips.plots.ImagePalette
    handler: python
    options:
    options:
      members:
        - close
        - save
        - draw_colored_disk
        - draw_grayscale_disk
        - ovearlay_localized_regions
        - plot_binary_localized_maps
        - draw_colored_synoptic_map
        - ovearlay_localized_synoptic_regions
        - plot_binary_localized_synoptic_maps
        - annotate
      show_root_heading: true
      show_source: false

::: chips.plots.ChipsPlotter
    handler: python
    options:
    options:
      members:
        - create_diagonestics_plots
        - create_output_stack
      show_root_heading: true
      show_source: false

::: chips.plots.SynopticChipsPlotter
    handler: python
    options:
    options:
      members:
        - create_synoptic_diagonestics_plots
        - create_synoptic_output_stack
      show_root_heading: true
      show_source: false