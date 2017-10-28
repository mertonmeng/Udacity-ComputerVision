function [H, theta, rho] = hough_lines_acc(BW, varargin)
    % Compute Hough accumulator array for finding lines.
    %
    % BW: Binary (black and white) image containing edge pixels
    % RhoResolution (optional): Difference between successive rho values, in pixels
    % Theta (optional): Vector of theta values to use, in degrees
    %
    % Please see the Matlab documentation for hough():
    % http://www.mathworks.com/help/images/ref/hough.html
    % Your code should imitate the Matlab implementation.
    %
    % Pay close attention to the coordinate system specified in the assignment.
    % Note: Rows of H should correspond to values of rho, columns those of theta.

    %% Parse input arguments
    p = inputParser;
    p.addParameter('RhoResolution', 1);
    p.addParameter('Theta', linspace(-90, 89, 180));
    p.parse(varargin{:});

    rhoStep = p.Results.RhoResolution;
    theta = p.Results.Theta;
    
    %% TODO: Your code here
    D = sqrt((size(BW,1) - 1)^2 + (size(BW,2) - 1)^2);
    nrho = 2*(ceil(D/rhoStep)) + 1;
    ntheta = length(theta);
    
    H = zeros(nrho,ntheta);
    diagonal = rhoStep*ceil(D/rhoStep);
    rho = -diagonal:rhoStep:diagonal;
    
    [r c] = find(BW > 0);
    
    for i = 1:size(r,1);
      for k = 1:ntheta
        a = theta(k);
        d = int32((c(i)*cos(a*pi/180) + r(i)*sin(a*pi/180) + diagonal)/rhoStep);
        H(d,a + 91) = H(d,a + 91) + 1;
      endfor
    endfor

endfunction
