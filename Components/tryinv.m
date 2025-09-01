function Ainv = tryinv(A)
    try
        Ainv = inv(A);
    catch ME
        if strcmp(ME.identifier, 'MATLAB:nearlySingularMatrix')
            Ainv = pinv(A);
%         else
%             disp(ME)
        end
    end
end