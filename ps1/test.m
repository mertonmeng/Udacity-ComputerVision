% ps1
pkg load image;  % Octave only

%% 1-a
img = imread(fullfile('input', 'ps1-input1.png'));  % already grayscale
%imshow(img);
img = rgb2gray(img);
%imshow(img);
h = fspecial('gaussian', 21, 5);
filteredImg = imfilter(img, h, 'symmetric');
%imshow(filteredImg);
%imwrite(filteredImg, fullfile('output', 'ps1-5-a-1.png'));

img_edges = edge(img, 'Canny');
img_edges_filtered = edge(filteredImg, 'Canny');
%imshow(img_edges);
%imshow(img_edges_filtered);
%imwrite(img_edges_filtered, fullfile('output', 'ps1-5-a-2.png'));
%figure;

radius = 20;

H1 = hough_circles_acc(img_edges_filtered, 20);
H2 = hough_circles_acc(img_edges_filtered, 23);

Himg = mat2gray(H1 + H2);

imshow(Himg, [0,1]);

hold("on");
imshow(img);
for i = 1:size(peaks,1)
  t = linspace(0,2*pi,100)'; 
  circsx = radius.*cos(t) + peaks(i,2) - 1; 
  circsy = radius.*sin(t) + peaks(i,1) - 1; 
  plot(circsx,circsy, "g");
endfor

saveas(2, fullfile('output', 'ps1-5-a-3.png'), "png");