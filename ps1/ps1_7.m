% ps1
pkg load image;  % Octave only

%% 1-a
img = imread(fullfile('input', 'ps1-input2.png'));  % already grayscale
%imshow(img);
img = rgb2gray(img);
imshow(img);
h = fspecial('gaussian', 17, 4);
filteredImg = imfilter(img, h, 'symmetric');
%imshow(filteredImg);
%imwrite(filteredImg, fullfile('output', 'ps1-5-a-1.png'));
figure;
%img_edges = edge(img, 'Canny');
img_edges_filtered = edge(filteredImg, 'Canny');

imshow(img_edges_filtered);
figure;

H = hough_circles_acc(img_edges_filtered, 25);

Himg = mat2gray(H);

imshow(Himg, [0,1]);
figure;
peaks = hough_peaks(H, 10);
hold("on");
imshow(img);
for i = 1:size(peaks,1)
  t = linspace(0,2*pi,100)'; 
  circsx = radius.*cos(t) + peaks(i,2) - 1; 
  circsy = radius.*sin(t) + peaks(i,1) - 1; 
  plot(circsx,circsy, "g");
endfor