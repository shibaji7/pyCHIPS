<!-- 
Author(s): Shibaji Chakraborty

Disclaimer:
pyCHIPS is under the MIT license found in the root directory LICENSE.md 
Everyone is permitted to copy and distribute verbatim copies of this license 
document.

This version of the MIT Public License incorporates the terms
and conditions of MIT General Public License.
-->
Data archived in the Joint Science Operations Center (JSOC) can be accessed through various methods. One way is to use Lookdata, a web-based tool that allows browsing of metadata, data series names, and generating export requests. Another method is to utilize the Python module called `drms`, which provides access to JSOC data through Python programming.

For direct data export, the web-based tool Exportdata can be used. It is particularly useful for users who already have a good understanding of the available data and the specific subsets they are interested in exporting from the JSOC.

IDL SolarSoft can also be used to access JSOC data, and the interface is described on the SolarSoft website. Additionally, users can access JSOC data through remote DRMS/SUMS sites by specifying the appropriate keyword in the IDL `vso_get` command. European users may refer to the Wissdom website for further information.

Users can also access commonly requested JSOC data through the Virtual Solar Observatory (VSO) site. Additionally, a sample script is provided for exporting recordsets using a local script for automation purposes. For those interested in becoming a `NetDRMS` service host, it is possible to have a personalized Data Records Management System development environment with automated data delivery. However, this process requires some involvement and further details can be found in the `NetDRMS` documentation.

For more details please visit JSOC's [How to get data](http://jsoc.stanford.edu/How_toget_data.html) page. Other information listed in their [home page](http://jsoc.stanford.edu/).