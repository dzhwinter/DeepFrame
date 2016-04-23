var img_path;
d3.selectAll( ".node" ).on( "mouseover", function(d) {
    img_path = "{{ image_path }}" + d.name + ".png";
    document.getElementById('thumb').src=img_path;
    document.getElementById('thumb').style.display='block';
})

