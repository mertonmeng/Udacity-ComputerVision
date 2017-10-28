% ps1
pkg load image;  % Octave only

%% 1-a
img = imread(fullfile('input', 'ps1-input0.png'));  % already grayscale
%% TODO: Compute edge image img_edges
img_edges = edge(img, 'Canny');
%imshow(img_edges);
%imwrite(img_edges, fullfile('output', 'ps1-1-a-1.png'));  % save as output/ps1-1-a-1.png

%% 2-a
[H, theta, rho] = hough_lines_acc(img_edges);  % defined in hough_lines_acc.m
%H = houghtf(img_edges);
Himg = mat2gray(H);

%[row col] = immaximas(H,1,100)


imshow(Himg, [0,1]);

%% TODO: Plot/show accumulator array H, save as output/ps1-2-a-1.png

%% 2-b
peaks = hough_peaks(H, 10)  % defined in hough_peaks.m
%% TODO: Highlight peak locations on accumulator array, save as output/ps1-2-b-1.png

for i = 1:size(peaks,1)
  rectangle("Position", [round(peaks(i,2) - 5) round(peaks(i,1) - 5) 10 10], "EdgeColor", 'green');
endfor
%print -dpng -color ps1-2-b-1.png
figure;
%% TODO: Rest of your code here
hough_lines_draw(img, "ps1-2-c-1.png", peaks, rho, theta);
