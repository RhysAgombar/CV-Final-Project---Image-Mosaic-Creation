# CV-Final-Project---Image-Mosaic-Creation
Image Mosaic Creation Program, by Rhys Agombar

Final project for Computer Vision course. The program consists of two implementations: static and dynamically sized mosaic modes. 

Both mosaic programs work by analyzing the texture and colour of segments of the image, and then finding similar matches in a provided image set. The textures were analyzed by using an LM Filter Bank, and the colour was chosen by finding the most prevalent colours in the images and comparing them to find the closest match. For simplicities sake, I used the [Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) for image matching. 

### Static Mosaic Examples
Static Mosaics use the same sized image segments for analysis across the entire image.

![Static Mosaic 1](/Result2.JPG)
![Static Mosaic 2](/Result3.JPG)

### Dynamic Mosaic Examples
Dynamic Mosaics use varying sizes of image segments when constructing the mosaic. The idea behind this was to replace large, low detail sections of an image with a single dataset image, and more detailed sections with smaller dataset images. The detail was detected using a simple convolution of a corner detecting filter.

![Static Mosaic 2](/variableMosaic3.JPG)
