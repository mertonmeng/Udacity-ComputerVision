pkg load image;

%% Load and convert image to double type, range [0, 1] for convenience
img_pepper = imread('pepper.png');
img_lenna = imread('lena.png');

noise = randn([size(img_pepper,1) size(img_pepper,2)])*25;

img_pepper(:,:,3) = img_pepper(:,:,3) + noise;
imshow(img_pepper);

%img_pepper = img_pepper - avg;
%img_pepper = img_pepper / stddev;
%img_pepper = img_pepper*10 + avg;
%imshow(img_pepper);