3D Backbone Interconnectivity in Semiconducting Polymer films
===================================

Poster Extension for **GRC: Electronic Processes in Organic Materials**


.. note::

   This project is under active development.

Interactive visualization
---------------------------

The embedded visualization may take up to 2 minutes to load.
Resize iframe after loading

.. raw:: html

   <script type="application/javascript">

   function resizeIFrameToFitContent( iFrame ) {

      iFrame.width  = iFrame.contentWindow.document.body.scrollWidth;
      iFrame.height = iFrame.contentWindow.document.body.scrollHeight;
   }

   window.addEventListener('DOMContentLoaded', function(e) {

      var iFrame = document.getElementById( 'plotly3D' );
      resizeIFrameToFitContent( iFrame );

      // or, to resize all iframes:
      var iframes = document.querySelectorAll("iframe");
      for( var i = 0; i < iframes.length; i++) {
         resizeIFrameToFitContent( iframes[i] );
      }
   } );

   </script>

   <iframe src="_static/S0_corner_streamplot1.html" id="plotly3D"></iframe>

