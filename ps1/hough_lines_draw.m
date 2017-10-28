function hough_lines_draw(img, outfile, peaks, rho, theta)
    % Draw lines found in an image using Hough transform.
    %
    % img: Image on top of which to draw lines
    % outfile: Output image filename to save plot as
    % peaks: Qx2 matrix containing row, column indices of the Q peaks found in accumulator
    % rho: Vector of rho values, in pixels
    % theta: Vector of theta values, in degrees

    % TODO: Your code here
    hold ("on");
    imshow(img);
    for i = 1:size(peaks, 1)
      d = rho(round(peaks(i,1)));
      a = theta(round(peaks(i,2)))*(pi/180);
      if(sin(a) == 0)
        x0 = d;
        y0 = 0;
        
        x1 = d;
        y1 = size(img,1);
      else
        x0 = 0;
        y0 = d/sin(a);
        
        x1 = size(img,2);
        y1 = (d - size(img,2)*cos(a))/sin(a);
      endif
      plot([x0 + 1 x1 + 1],[y0 - 1 y1 - 1],"g");
    endfor
    saveas(3, fullfile('output', outfile), "png");
    
endfunction
