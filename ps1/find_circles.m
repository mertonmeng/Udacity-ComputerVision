function [centers, radii] = find_circles(BW, radius_range)
    % Find circles in given radius range using Hough transform.
    %
    % BW: Binary (black and white) image containing edge pixels
    % radius_range: Range of circle radii [min max] to look for, in pixels

    % TODO: Your code here
    
    totalH = zeros(size(BW));
    
    [row col] = find(BW > 0);
    centers = [];
    radii = [];
   
    for r = radius_range(1):radius_range(2)
      H = zeros(size(BW));
      for i = 1:size(row,1)
        for theta = 0:2:359
          w = theta*(pi/180);
          a = round(col(i) - r*cos(w));
          b = round(row(i) + r*sin(w));
          if (a < 0 || a > size(BW,2) - 1 || b < 0 || b > size(BW,1) - 1)
            continue
          else
            H(b + 1, a + 1) = H(b + 1, a + 1) + 1;
          endif
        endfor
      endfor  
      peaks = hough_peaks(totalH, 10);
      centers = [centers; peaks];
      radii = [radii; r.*ones(size(peaks,1),1)];
    endfor
    
    %peaks = hough_peaks(totalH, 10);  % defined in hough_peaks.m
    
    
endfunction
