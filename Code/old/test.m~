
[x,y]=ndgrid(0:20,0:30);
I=imread('media/Baby/img/0001.jpg');
imshow(I);
I=rgb2gray(I)
I=double(I)*(1/255);

sigma=3;

[Ix,Iy]=image_derivatives(I,sigma)



function [Ix,Iy]=image_derivatives(I,sigma)
% Make derivatives kernels
[x,y]=ndgrid(floor(-3*sigma):ceil(3*sigma),floor(-3*sigma):ceil(3*sigma));
DGaussx=-(x./(2*pi*sigma^4)).*exp(-(x.^2+y.^2)/(2*sigma^2));
DGaussy=-(y./(2*pi*sigma^4)).*exp(-(x.^2+y.^2)/(2*sigma^2));
% Filter the images to get the derivatives
Ix = imfilter(I,DGaussx,'conv');
Iy = imfilter(I,DGaussy,'conv');
end

