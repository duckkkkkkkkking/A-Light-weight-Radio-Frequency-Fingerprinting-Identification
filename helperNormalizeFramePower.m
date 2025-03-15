function y = helperNormalizeFramePower(x)
%helperNormalizeFramePower Normalize frame power
%   Y = helperNormalizeFramePower(X) normalizes the signal power of input,
%   X, and returns the normalized frame, Y.

%   Copyright 2019 The MathWorks, Inc.

framesPower = mean(abs(x).^2);
y = x./sqrt(framesPower);

