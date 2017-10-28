function peaks = hough_peaks(H, varargin)
    % Find peaks in a Hough accumulator array.
    %
    % Threshold (optional): Threshold at which values of H are considered to be peaks
    % NHoodSize (optional): Size of the suppression neighborhood, [M N]
    %
    % Please see the Matlab documentation for houghpeaks():
    % http://www.mathworks.com/help/images/ref/houghpeaks.html
    % Your code should imitate the matlab implementation.

    %% Parse input arguments
    p = inputParser;
    p.addOptional('numpeaks', 1, @isnumeric);
    p.addParamValue('Threshold', 0.5 * max(H(:)));
    p.addParamValue('NHoodSize', floor(size(H) / 100.0) * 2 + 1);  % odd values >= size(H)/50
    p.parse(varargin{:});

    numpeaks = p.Results.numpeaks;
    threshold = p.Results.Threshold;
    nHoodSize = p.Results.NHoodSize;

    % TODO: Your code here
    halfWinWidth = int32((nHoodSize(1) - 1)/2);
    dummyH = zeros(size(H,1) + 2*halfWinWidth, size(H,2) + 2*halfWinWidth);
    dummyH(halfWinWidth + 1 : size(dummyH,1) - halfWinWidth, halfWinWidth + 1 : size(dummyH,2) - halfWinWidth) = H;
    minIdx = 1;
    peaksCell = {};
    
    for i = halfWinWidth + 1 : size(dummyH,1) - halfWinWidth
      for j = halfWinWidth + 1 : size(dummyH,2) - halfWinWidth
        patch = dummyH(i-halfWinWidth:i+halfWinWidth,j-halfWinWidth:j+halfWinWidth);
        maxVal = max(patch(:));
        
        if (maxVal < threshold)
          continue
        endif
        
        %maintain the peaks list
        if (dummyH(i,j) == maxVal)
          if (size(peaksCell,1) == 0)
            peaksCell(1, 1) = [i j];
            minIdx = 1;
          else if (size(peaksCell,1) > 0 && size(peaksCell,1)  < numpeaks)
            peaksCell(size(peaksCell,1) + 1, 1) = [i j];
            if (dummyH(i,j) < dummyH(peaksCell{minIdx, 1}(1), peaksCell{minIdx, 1}(2)))
              minIdx = size(peaksCell,1);
            endif
          else
            if (dummyH(i,j) > dummyH(peaksCell{minIdx,1}(1), peaksCell{minIdx,1}(2)))
              peaksCell(minIdx,1) = [i j];
              minInPeaks = 10000000;
              for k = 1:size(peaksCell,1)
                if (minInPeaks > dummyH(peaksCell{k,1}(1), peaksCell{k,1}(2)))
                  minInPeaks = dummyH(peaksCell{k,1}(1), peaksCell{k,1}(2));
                  minIdx = k;
                endif
              endfor
            endif
          endif
          
        endif
        endif
      endfor
    endfor
    
    for k = 1:size(peaksCell,1)
      peaksCell{k,1} = peaksCell{k,1} - halfWinWidth;
    endfor
    
    peaks = cell2mat(peaksCell);
    
endfunction
