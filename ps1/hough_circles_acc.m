function H = hough_circles_acc(BW, radius)
    % Compute Hough accumulator array for finding circles.
    %
    % BW: Binary (black and white) image containing edge pixels
    % radius: Radius of circles to look for, in pixels

    % TODO: Your code here
    
    H = zeros(size(BW));
    
    [row col] = find(BW > 0);
   
    for i = 1:size(row,1)
      for theta = 0:359
        w = theta*(pi/180);
        a = round(col(i) - radius*cos(w));
        b = round(row(i) + radius*sin(w));
        if (a < 0 || a > size(BW,2) - 1 || b < 0 || b > size(BW,1) - 1)
          continue
        else
          H(b + 1, a + 1) = H(b + 1, a + 1) + 1;
        endif
      endfor
    endfor  
    
endfunction
