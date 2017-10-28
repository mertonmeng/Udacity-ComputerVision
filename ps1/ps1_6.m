% ps1
pkg load image;  % Octave only

%% 1-a
img = imread(fullfile('input', 'ps1-input2.png'));  % already grayscale
%imshow(img);
img = rgb2gray(img);
%imshow(img);
%figure;
h = fspecial('gaussian', 29, 7);
filteredImg = imfilter(img, h, 'symmetric');
%imshow(filteredImg);

img_edges = edge(img, 'Canny');
img_edges_filtered = edge(filteredImg, 'Canny', [0.6 0.7]);
%imshow(img_edges);
imshow(img_edges_filtered);
figure;

[H, theta, rho] = hough_lines_acc(img_edges_filtered, 'RhoResolution', 2);
Himg = mat2gray(H);
imshow(Himg, [0,1]);
%imwrite(Himg, fullfile('output', 'ps1-6-test.png'));

peaks = hough_peaks(H, 4, 'NHoodSize', 8, 'Threshold', 0.6 * max(H(:)));  % defined in hough_peaks.m
%for i = 1:size(peaks,1)
%  rectangle("Position", [round(peaks(i,2) - 5) round(peaks(i,1) - 5) 10 10], "EdgeColor", 'green');
%endfor
%saveas(2, fullfile('output', 'ps1-4-c-1.png'), "png");
figure;
hough_lines_draw(img, "ps1-6-c-1.png", peaks, rho, theta);