% ps1
pkg load image;  % Octave only

%% 1-a
img = imread(fullfile('input', 'ps1-input0-noise.png'));  % already grayscale

%imshow(img);
h = fspecial('gaussian', 25, 6);
filteredImg = imfilter(img, h, 'symmetric');
%imshow(filteredImg);
%imwrite(filteredImg, fullfile('output', 'ps1-3-a-1.png'));
img_edges = edge(img, 'Canny');
img_edges_filtered = edge(filteredImg, 'Canny');
imshow(img_edges_filtered);
%imwrite(img_edges, fullfile('output', 'ps1-3-b-1.png'));
imwrite(img_edges_filtered, fullfile('output', 'ps1-3-b-2.png'));
figure;
[H, theta, rho] = hough_lines_acc(img_edges_filtered);  % defined in hough_lines_acc.m
%H = houghtf(img_edges);
Himg = mat2gray(H);

imshow(Himg, [0,1]);

%% TODO: Plot/show accumulator array H, save as output/ps1-2-a-1.png

%% 2-b
peaks = hough_peaks(H, 10)  % defined in hough_peaks.m
for i = 1:size(peaks,1)
  rectangle("Position", [round(peaks(i,2) - 5) round(peaks(i,1) - 5) 10 10], "EdgeColor", 'green');
endfor
saveas(3, fullfile('output', 'ps1-3-c-1.png'), "png");
figure;
hough_lines_draw(img, "ps1-3-c-2.png", peaks, rho, theta);