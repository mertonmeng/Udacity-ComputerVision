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

[centers radius] = find_circles(img_edges_filtered, [20 23]);

