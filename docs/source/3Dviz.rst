3D Backbone Interconnectivity of Semiconducting Polymers
===================================

Poster Extension for *3D Liquid Crystalline Connectivity in Semiconducting
Polymer Films* poster, presented at the **Gordon Research Conference: 
Electronic Processes in Organic Materials**

Charge transport pathways that follow polymer backbone connectivity are visualized 
as streamlines generated from the director field of backbone orientations. The black 
dots indicate the location of seed points for each set of streamlines, and each set of 
streamlines has been plotted with a different color. 

In this plot, we examine the backbone transport pathways in a 13nm x 30 nm x 5 nm volume of the film.
We have initialized streamlines from different vertical planes, spaced 4.3 nm apart.
Starting from the seed points, streamlines are drawn incrementally, following the  local director field. 
The radius of the streamlines correspond to the local divergence of the director field.

The visualization illustrates that semiconducting polymer films have heterogenous charge 
transport. For example, if charges were injected at the plane at 82 nm (pink) and collected
at 95 nm (purple), the charges would follow 3D backbone paths that largely go around the bulk
of this volume, rather than the shortest possible path through the volume.


Interactive visualization
---------------------------

The embedded visualization may take up to 2 minutes to load. 
The plot is rendered using plotly backend. 
The buttons on the left side will toggle the visibility of streamtubes that were
initialized at the labelled y-value, in Angstroms.
Toggling the visiblity may take up to 30 seconds for the plotly.js backend to reload.
 Performance speedups will be included in future versions.

.. note::

   Zoom features unavailable on mobile devices



.. raw:: html

   <iframe src="_static/S0_corner_streamplot_820to950by63.html" id="plotly3D" style="border:none; width: 100%; height: 100vh"></iframe>




