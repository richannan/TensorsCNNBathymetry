function dd_normalized = fNormalizeData(matrix)
% Funtion to normalize dataset so that the it ranges between 0 and 1

%By:    Richard Fiifi ANNAN (richannan@outlook.com)
%       School of Land Science and Technology
%       China University of Geosciences (Beijing)


assert(isnumeric(matrix), "Input must be numeric")
assert(ismatrix(matrix) || isvector(matrix), "Input must be a matrix or a vector");


dd = matrix;

        if (min(dd(:)) < 0) && (max(dd(:)) <= 0)
            dd1 = dd * -1;
            dd_normalized = dd1 / max(dd1(:)); 
        elseif min(dd(:)) < 0 && max(dd(:)) > 0
            dd1 = abs(min(dd(:)) - max(dd(:)));
            dd2 = dd + dd1;
            dd_normalized = dd2 / max(dd2(:));
        else
            dd_normalized = dd / max(dd(:));
        end

end